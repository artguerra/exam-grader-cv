import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from cv2.typing import MatLike

def separate_questions(image: MatLike)-> Tuple[MatLike, List[Tuple[int, int, int, int]]]:
    """
    Divide the answer sheet into different question boxes
    """
    # convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # convert image to black and white (inverted)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find the outermost contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    question_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # big wide rectangles
        if area > 1000 and w > 100:
            question_boxes.append((x, y, w, h - 25)) # -25 is temporary fix for now
    question_boxes.sort(key = lambda b:b[1])

    return (thresh, question_boxes)

def detect_all_bubbles(thresh: MatLike, box: Tuple[int, int, int, int]):
    """
    detect the bubble in the question
    """
    x, y, w, h = box
    # MCQ area
    mcq = thresh[y:y+h, x:x+w].astype(np.uint8)
    # find the all bubble contours
    bubble_cnts, _ = cv2.findContours(
        mcq,
        cv2.RETR_CCOMP, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    # print(bubble_cnts)
    bubbles = []
    for c in bubble_cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        # print(bx, by, bw, bh)
        # filter by bubble size
        if 20 < bw < 60 and 20 < bh < 60:
            bubbles.append((bx, by, bw, bh))
    # print(bubbles)
    # sort from left to right, top to bottom
    bubbles = sorted(bubbles, key = lambda b: b[0])
    
    return (mcq, bubbles)

def detect_filled_bubbles(mcq: MatLike, bubbles: List):
    """
    Given the cropped question image and its bubble boxes,
    return list of filled bubble indices and debug masks.
    """
    filled_index = []
    for idx, (bx, by, bw, bh) in enumerate(bubbles):
        # create a mask
        mask = np.zeros(mcq.shape, dtype=np.uint8)
        # fill bubble bouding box with white on mask
        cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)
        # count white pixels in the bubble
        filled_pixels = cv2.countNonZero(mask & mcq)
        # check if bubble fill exceeds threshold
        if filled_pixels > 0.6 * (bw * bh): # TODO: adjust the threshold
            filled_index.append(idx)
    # return answers and the per-question debug masks dictionary
    return filled_index

def MCQ_box(thresh: MatLike, box: Tuple[int, int, int, int]):
    """
    For mcq questions, call this function
    """
    x, y, w, h = box
    mcq = thresh[y:y+h, x:x+w].astype(np.uint8)
    mcq, bubbles = detect_all_bubbles(thresh, box)
    answers, _ = detect_filled_bubbles(mcq, bubbles)
    return answers

