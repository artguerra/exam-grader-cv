import argparse
from typing import cast

import cv2

from preprocess import detect_page_mask
from rectify import rectify_page
from process_question import separate_questions, detect_all_bubbles, detect_filled_bubbles


def grading_pipeline(path: str):
    img = cv2.imread(path)

    assert img is not None

    page_mask, page_bbox = detect_page_mask(img)
    # crop image based on circle position for now
    crop = rectify_page(img, page_mask, page_bbox)
    thresh, question_boxes = separate_questions(crop)
    # visualize
    for (x, y, w, h) in question_boxes:
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for i in range(0, len(question_boxes)):
        bubbles = detect_all_bubbles(thresh, question_boxes[i]) 
        for (bx, by, bw, bh) in bubbles:
            cv2.rectangle(thresh, (bx, by), (bx + bw, by + bh), 255, -1)
    cv2.imshow("TEST", crop)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.filename))
