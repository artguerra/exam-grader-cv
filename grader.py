import argparse
import json
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob
from typing import cast

import cv2
import zxingcpp
from PIL import Image
from cv2.typing import MatLike
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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


def grading_pipeline(path: str, user_dpi: int | None):
    # global variables
    global student_id
    global student_recognized
    global exam
    global qrcode_data

    image_paths = sorted(glob.glob(path + "*"))
    exam_images = [cv2.imread(p) for p in image_paths]
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )

    output_pages = []
    pages_thresh = []
    question_boxes = []

    for i, (img_path, img) in enumerate(zip(image_paths, exam_images)):
        assert img is not None

        if user_dpi is not None:
            dpi = user_dpi
        else:
            dpi = get_image_dpi(img_path)

            if dpi is None:
                raise Exception("Could not identify image DPI")

            dpi = dpi[0] # take x coord dpi

        print(f"Image DPI: {dpi}")

        # document isolation (preprocessing)
        page_mask, mask_centroid = detect_page_mask(img)

        # correct image perspective based on fiducials positions
        warped = rectify_page(img, page_mask, mask_centroid, dpi)
        output_pages.append(warped.copy())

        if i == 0:
            # identify exam and variant by the qrcode
            identified_qrcodes = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.QRCode)

            if not identified_qrcodes:
                raise Exception("Could not identify exam data (qrcode not found)")

            qrcode_data_str = xor_decrypt_from_hex(identified_qrcodes[0].text, SECRET_KEY)
            qrcode_data = json.loads(qrcode_data_str)

            exam = json.load(open(f"exams/exam_{qrcode_data['exam_id']}.json"))

        # find question boxes of the page
        thresh, page_question_boxes = separate_questions(warped, dpi)
        print(f"question boxes on page {i}: {len(page_question_boxes)}")
        pages_thresh.append(thresh)
        question_boxes.extend([(box, i) for box in page_question_boxes])

    order = exam["variant_ordering"][qrcode_data["variant"]]
    q_by_index = {q["index"]: q for q in exam["questions"]}

    # treat first question box (student id)
    student_id_box, _ = question_boxes[0]
    student_id, _ = numeric_box(pages_thresh[0], student_id_box, processor, model)
    student_recognized = False

    try:
        if not student_id:
            raise ValueError()

        student_id = int(student_id[0].replace(' ', '').replace('.', ''))
        student_recognized = True
        print(f"Student id: {student_id}")
    except ValueError:
        print("Student could not be identified.")

    for question_idx, (box, page_idx) in zip(order, question_boxes[1:]):
        q = q_by_index[question_idx]

        if q["type"] == "MCQ":
            answer_idx, bubbles = MCQ_box(pages_thresh[page_idx], box, dpi)
            answer = [chr(c + ord("A")) for c in answer_idx]

            print(
                f"question {question_idx}. answer was: {answer}, correct answer is: {q['correct']}"
            )

            if answer != q["correct"]:
                for opt in q["correct"]:
                    idx = ord(opt) - ord("A")
                    draw_cross(output_pages[page_idx], bubbles[idx])

        elif q["type"] == "NUM":
            answer, global_pos = numeric_box(pages_thresh[page_idx], box, processor, model)
            student_ans_str = "".join(answer)
            correct_val = q["correct"]
            tolerance = q["tolerance"]
            
            print(f"question {question_idx}. answer was: {student_ans_str}, correct answer is: {correct_val}")

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
                    output_pages[page_idx],
                    str(correct_val),
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )

    # export result
    folder = f"graded_exam_{exam['exam_id']}"
    os.makedirs(folder, exist_ok=True)

    student_text = student_id if student_recognized else "UNKNOWN"

    for i, page_img in enumerate(output_pages):
        filename = f"graded_{qrcode_data['exam_id']}_page{i + 1}_{student_text}.jpg"
        full_path = os.path.join(folder, filename)
        
        cv2.imwrite(full_path, page_img)
        print(f"Saved page {i+1} to {full_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("directory", type=str)
    parser.add_argument("--dpi", type=int, help="Manually set DPI to be used (dont auto-detect)")

    args = parser.parse_args()

    grading_pipeline(cast(str, args.directory), args.dpi)
