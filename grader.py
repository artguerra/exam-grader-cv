import argparse
from typing import cast

import cv2
import json

from preprocess import detect_page_mask
from process_question import MCQ_box, separate_questions, ocr_numeric_question
from rectify import rectify_page
from process_question import separate_questions, detect_all_bubbles, detect_filled_bubbles, MCQ_box


def grading_pipeline(path: str):
    img = cv2.imread(path)

    assert img is not None

    page_mask, page_bbox = detect_page_mask(img)
    # crop image based on circle position for now
    crop = rectify_page(img, page_mask, page_bbox)
    thresh, question_boxes = separate_questions(crop)

    order = exam["variant_ordering"][barcode_data["variant"]]
    q_by_index = {q["index"]: q for q in exam["questions"]}

    # grades = []

    for question_idx, box in zip(order, question_boxes):
        q = q_by_index[question_idx]

        if q["type"] == "MCQ":
            answer = MCQ_box(thresh, box)
            answer = [chr(c + ord('A')) for c in answer]

            print(f"question {question_idx}. answer was: {answer}, correct answer is: {q['correct']}")

            # if q["correct"] == answer:
            #     print(f"question {question_idx} is correct!")
            # else:
            #     print(f"question {question_idx} is incorrect!")
            # compare with q["correct"]
        elif q["type"] == "NUM":
            answer, ocr_area = ocr_numeric_question(thresh, box)
            print(f"question {question_idx}. answer was: {answer}, correct answer is: {q['correct']}")
            if q["correct"] == answer:
                print(f"question {question_idx} is correct!")
            else:
                print(f"question {question_idx} is incorrect!")
            cv2.imshow("test", ocr_area)
            cv2.waitKey(0)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.filename))
