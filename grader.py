import argparse
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from typing import cast

import cv2
import zxingcpp
from PIL import Image
from cv2.typing import MatLike
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import glob

from exam import Exam
from generator import SECRET_KEY
from preprocess import detect_page_mask
from process_question import MCQ_box, numeric_box, separate_questions
from rectify import rectify_page
from util import xor_decrypt_from_hex

def get_image_dpi(path: str):
    with Image.open(path) as img:
        dpi = img.info.get('dpi')
        
        return dpi


def draw_cross(img: MatLike, box: tuple[int, int, int, int]):
    """Draws a X inside the bounding box."""
    # thickness and color
    color = (0, 0, 255)
    thickness = 2
    x, y, w, h = box

    # draw two diagonal lines
    cv2.line(img, (x, y), (x + w, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x + w, y), color, thickness)


def grading_pipeline(path: str):
    # global variables
    global student_id
    global student_recognized
    global exam
    global barcode_data
    image_paths = sorted(glob.glob(path + "*"))
    exam_images = [cv2.imread(p) for p in image_paths]
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )
    #img = cv2.imread(path)
    page_number = 0 # 0 is page 1 actually
    question_offset = 0
    for img_path, img in zip(image_paths, exam_images):   
        assert img is not None
        dpi = get_image_dpi(img_path)
        assert dpi is not None
        dpi = dpi[0] # take x coord dpi
        #print(f"Image DPI: {dpi}")

        page_mask, page_bbox = detect_page_mask(img)

        # warped image based on circle position for now
        warped = rectify_page(img, page_mask, page_bbox, dpi)
        output = warped.copy()
        if page_number == 0:
            # identify exam and variant by the barcode
            # identified_barcodes = zxingcpp.read_barcodes(img)

            # if not identified_barcodes:
            #     raise Exception("Could not identify exam data (barcode not found)")
            # print(identified_barcodes[0].text)
            # barcode_data_str = xor_decrypt_from_hex(identified_barcodes[0].text, SECRET_KEY)
            # barcode_data = json.loads(barcode_data_str)
            barcode_data = {
                "exam_id": "E01",
                "variant": 0
            }
            exam = json.load(open(f"exams/exam_{barcode_data['exam_id']}.json"))

        # find question boxes
        thresh, question_boxes = separate_questions(warped, dpi)
        order = exam["variant_ordering"][barcode_data["variant"]]
        q_by_index = {q["index"]: q for q in exam["questions"]}

        #for question_idx, box in zip(order, question_boxes):
        for i, box in enumerate(question_boxes):
            
            if page_number == 0 and i == 0: # the first box on the first page
                # treat first question box (student id)
                student_id_box = question_boxes[0]
                student_id, _ = numeric_box(thresh, student_id_box, processor, model)
                student_recognized = False

                try:
                    student_id = int(student_id[0].replace(' ', '').replace('.', ''))
                    student_recognized = True
                    print(f"Student id: {student_id}")
                except ValueError:
                    print("Student could not be identified.")
                continue # no question detection for this box
            
            q = q_by_index[i + question_offset] # i will start with 1

            if q["type"] == "MCQ":
                answer_idx, bubbles = MCQ_box(thresh, box, dpi)
                answer = [chr(c + ord("A")) for c in answer_idx]

                print(
                    f"question {i + question_offset}. answer was: {answer}, correct answer is: {q['correct']}"
                )

                if answer != q["correct"]:
                    for opt in q["correct"]:
                        idx = ord(opt) - ord("A")
                        draw_cross(output, bubbles[idx])
            elif q["type"] == "NUM":
                answer, global_pos = numeric_box(thresh, box, processor, model)
                student_ans_str = "".join(answer)
                correct_val = q["correct"]
                tolerance = q["tolerance"]
                
                print(f"question {i + question_offset}. answer was: {student_ans_str}, correct answer is: {correct_val}")

                # determine if incorrect
                is_correct = False
                try:
                    if student_ans_str and abs(float(student_ans_str) - float(correct_val)) <= tolerance:
                        is_correct = True
                except ValueError:
                    pass # conversion failed (empty string or garbage), count as wrong

                # if wrong, write the correct answer
                if not is_correct:
                    # write to the right of the box
                    right_edge = max(gx + gw for gx, gy, gw, gh in global_pos)

                    # center vertically relative to the boxes
                    avg_y = int(sum(gy + gh // 2 for gx, gy, gw, gh in global_pos) / len(global_pos))
                    text_x = right_edge + 120
                    text_y = avg_y + 10

                    cv2.putText(
                        output,
                        str(correct_val),
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )
        # --- EXPORT RESULT ---
        folder = "graded_exam"
        os.makedirs(folder, exist_ok=True)
        student_text = student_id if student_recognized else "UNKNOWN"
        output_filename = f"/{folder}/graded_P{page_number + 1}/graded_{barcode_data['exam_id']}_{student_text}.jpg"
        cv2.imwrite(output_filename, output)
        print(f"Grading complete. Saved correction to {output_filename}")
        page_number += 1 # increment page number
        question_offset += len(question_boxes) # increment question offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("directory", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.directory))
