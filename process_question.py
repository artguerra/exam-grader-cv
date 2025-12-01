import cv2
import numpy as np
from scipy.signal import find_peaks
import numpy.typing as npt
from typing import Tuple, List
from cv2.typing import MatLike
from keras.models import load_model

from generator import BUBBLE_RADIUS, FIDUCIAL_OFFSET, GUTTER, MARGIN, PT_TO_PX

QUESTION_BOX_OFFSET = (MARGIN + GUTTER - FIDUCIAL_OFFSET) * PT_TO_PX
BUBBLE_RADIUS_PX = BUBBLE_RADIUS * PT_TO_PX


from generator import BUBBLE_RADIUS, FIDUCIAL_OFFSET, GUTTER, MARGIN, PT_TO_PX

QUESTION_BOX_OFFSET = (MARGIN + GUTTER - FIDUCIAL_OFFSET) * PT_TO_PX
BUBBLE_RADIUS_PX = BUBBLE_RADIUS * PT_TO_PX


def separate_questions(
    image: MatLike,
) -> tuple[MatLike, list[tuple[int, int, int, int]]]:
    """
    Divide the answer sheet into different question boxes
    """
    # convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert image to black and white (inverted)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find the outermost contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    question_boxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # only treat rectangles
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # filter rectangles by margin distance, as defined in the generator
        if abs(x - QUESTION_BOX_OFFSET) > 5:  # 5 px tolerance
            continue

        question_boxes.append((x, y, w, h))

    question_boxes.sort(key=lambda b: b[1])

    return (thresh, question_boxes)


def detect_all_bubbles(thresh: MatLike, box: tuple[int, int, int, int]):
    """
    detect the bubble in the question
    """
    x, y, w, h = box
    # MCQ area
    mcq = thresh[y : y + h, x : x + w].astype(np.uint8)
    mcq_blur = cv2.GaussianBlur(mcq, (5, 5), 0)

    circles = cv2.HoughCircles(
        mcq_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=2 * BUBBLE_RADIUS_PX,  # min distance between centers
        param1=50,
        param2=10,
        minRadius=int(BUBBLE_RADIUS_PX - 1),
        maxRadius=int(BUBBLE_RADIUS_PX + 1),
    )

    bubbles = []

    if circles is not None:
        circles = np.around(circles[0]).astype(np.uint16)  # x,y,r
        for cx, cy, r in circles:
            bx, by, bw, bh = cx - r, cy - r, 2 * r, 2 * r
            bubbles.append((bx, by, bw, bh))

    # sort from left to right, top to bottom
    bubbles.sort(key=lambda b: b[0])

    return mcq, bubbles


def detect_filled_bubbles(mcq: MatLike, bubbles: list):
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
        if filled_pixels > 0.6 * (bw * bh):  # TODO: adjust the threshold
            filled_index.append(idx)

    # return answers and the per-question debug masks dictionary
    return filled_index


def MCQ_box(thresh: MatLike, box: tuple[int, int, int, int]):
    """
    For mcq questions, call this function
    """
    x, y, w, h = box
    mcq = thresh[y : y + h, x : x + w].astype(np.uint8)
    mcq, bubbles = detect_all_bubbles(thresh, box)
    answers = detect_filled_bubbles(mcq, bubbles)
    return answers

def find_ocr_box(thresh: MatLike, box: tuple[int, int, int, int]):
    """
    find the area to fill in answer
    
    :param thresh: black and white image
    :type thresh: MatLike
    :param box: question box (x, y, w, h)
    :type box: tuple[int, int, int, int]
    """
    x, y, w, h = box
    roi = thresh[y : y + h, x : x + w].astype(np.uint8)
    gray_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    # Find contours inside the question box
    contours, _ = cv2.findContours(
        gray_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    ) 
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Ignore tiny noise
        if area < 50:
            continue
        if area > 40000 and area < 50000: # size is 43680
            rects.append((area, (x, y, w, h)))

    # Sort DESCENDING by area
    rects.sort(key=lambda r: r[0], reverse=True)
    x, y, w, h = rects[0][1]
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y+ h
    cropped_text = roi[y1:y2, x1:x2]
    return cropped_text
    

def ocr_numeric_question(thresh: MatLike, box: tuple[int, int, int, int]):
    """
    Call this function for numeric questions
    
    :param thresh: black and white image
    :type thresh: MatLike
    :param box: question box (x, y, w, h)
    :type box: tuple[int, int, int, int]
    """
    answer_area = find_ocr_box(thresh, box)
    # segment touching digits
    digit_rois = segment_digits_touching(answer_area)

    # load MNIST CNN
    model = load_model("cnn.h5")

    result_digits = []

    for roi in digit_rois:
        mnist_img = prep_for_mnist(roi)
        pred = np.argmax(model.predict(mnist_img)) # type: ignore
        result_digits.append(str(pred))

    return "".join(result_digits)
    
    
def prep_for_mnist(img):
    # crop whitespace
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # make square
    size = max(w, h)
    padded = np.full((size, size), 255, dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # resize to 28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # normalize
    arr = resized.astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def segment_digits_touching(thresh_img):
    # sum white pixels vertically
    projection = np.sum(255 - thresh_img, axis=0)

    # find splits between digits using valleys in projection
    peaks, _ = find_peaks(-projection, distance=10, prominence=50)

    # Ensure boundaries at image edges
    boundaries = [0] + peaks.tolist() + [thresh_img.shape[1]]

    # extract digit ROIs
    rois = []
    for i in range(len(boundaries) - 1):
        x1 = boundaries[i]
        x2 = boundaries[i+1]
        
        # ignore very small sections
        if x2 - x1 < 10:
            continue
        
        roi = thresh_img[:, x1:x2]
        rois.append(roi)

    return rois
    
    