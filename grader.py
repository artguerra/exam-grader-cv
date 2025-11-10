import argparse
from typing import cast

import cv2

from preprocess import detect_page_mask
from rectify import rectify_page


def grading_pipeline(path: str):
    img = cv2.imread(path)

    assert img is not None

    page_mask, page_bbox = detect_page_mask(img)
    rectify_page(img, page_mask, page_bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam grader", description="Grades an exam created in our auto exam format"
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    grading_pipeline(cast(str, args.filename))
