import argparse
from typing import cast

import cv2
import json

from preprocess import detect_page_mask
from rectify import rectify_page
from process_question import separate_questions, detect_all_bubbles, detect_filled_bubbles, MCQ_box


def grading_pipeline(path: str):
    img = cv2.imread(path)

    assert img is not None

    page_mask, page_bbox = detect_page_mask(img)

    # crop image based on circle position for now
    crop = rectify_page(img, page_mask, page_bbox)

    thresh, question_boxes = separate_questions(crop)

    # TODO: correspond questions to their question type
    # with open(json_path) as f:
    #     d = json.load(f)
    #     print(d)
   
    mcq, bubbles = detect_all_bubbles(thresh, question_boxes[0])
    bubbles_filled = detect_filled_bubbles(mcq, bubbles)
    
    answers = MCQ_box(thresh, question_boxes[0])
    print(answers)

    # visualize
    # vis = crop.copy()
    # 
    # # DEBUG for question boxes
    # for (x, y, w, h) in question_boxes:
    #     cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imshow("question boxes", cv2.resize(vis, (545, 842)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # DEBUG all bubbles in one question
    # for (bx, by, bw, bh) in bubbles:
        # cv2.rectangle(mcq_region, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

    # Debug MCQ region filled bubbles
    x, y, w, h = question_boxes[0]
    mcq_region = crop[y:y+h, x:x+w]
    for idx in bubbles_filled:
        bx, by, bw, bh = bubbles[idx]
        cv2.rectangle(mcq_region, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

    cv2.imshow("Painted in MCQ Area", mcq_region)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.filename))
