import argparse
import json
from typing import cast

import cv2
import zxingcpp

from exam import Exam
from generator import SECRET_KEY
from preprocess import detect_page_mask
from process_question import MCQ_box, separate_questions, numeric_box
from rectify import rectify_page
from util import xor_decrypt_from_hex
from tensorflow.keras.models import load_model


def grading_pipeline(path: str):
    img = cv2.imread(path)

    assert img is not None

    page_mask, page_bbox = detect_page_mask(img)

    # crop image based on circle position for now
    crop = rectify_page(img, page_mask, page_bbox)
    # identify exam and variant by the barcode
    identified_barcodes = zxingcpp.read_barcodes(img)

    if not identified_barcodes:
        raise Exception("Could not identify exam data (barcode not found)")

    barcode_data_str = xor_decrypt_from_hex(identified_barcodes[0].text, SECRET_KEY)
    barcode_data = json.loads(barcode_data_str)

    exam: Exam = json.load(open(f"exams/exam_{barcode_data['exam_id']}.json"))

    # find question boxes
    thresh, question_boxes = separate_questions(crop)
    # cv2.imshow("test", cv2.resize(thresh, (545, 842)))
    # cv2.waitKey(0)
    order = exam["variant_ordering"][barcode_data["variant"]]
    q_by_index = {q["index"]: q for q in exam["questions"]}

    # grades = []

    for question_idx, box in zip(order, question_boxes):
        q = q_by_index[question_idx]

        if q["type"] == "MCQ":
            answer = MCQ_box(thresh, box)
            answer = [chr(c + ord('A')) for c in answer]

            print(f"question {question_idx}. answer was: {answer}, correct answer is: {q['correct']}")
            # compare with q["correct"]
        elif q["type"] == "NUM":
            model = load_model("./model/cnn.h5", safe_mode=False)
            answer = numeric_box(thresh, box, model)
            print(f"question {question_idx}. answer was: {"".join(answer)}, correct answer is: {q['correct']}")
            # # debug the small rectangles
            # for (x, y, w, h) in global_pos:
            #     cv2.rectangle(crop, (x, y), (x + w, y + h), (0,255, 0), 5)
            
    cv2.imshow("test", cv2.resize(crop, (545, 842)))
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.filename))
