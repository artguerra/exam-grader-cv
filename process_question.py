import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image

from generator import BUBBLE_RADIUS, FIDUCIAL_OFFSET, GUTTER, MARGIN, PT_TO_PX

QUESTION_BOX_OFFSET = (MARGIN + GUTTER - FIDUCIAL_OFFSET) * PT_TO_PX
BUBBLE_RADIUS_PX = BUBBLE_RADIUS * PT_TO_PX


def debug_draw_boxes(img: MatLike, boxes: list[tuple[int, int, int, int]], title="Debug"):
    """
    Draws boxes on a copy of the image and displays it.
    """
    vis = img.copy()

    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # resize for better visibility
    cv2.imshow(title, cv2.resize(vis, (545, 842)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def separate_questions(
    image: MatLike,
) -> tuple[MatLike, list[tuple[int, int, int, int]]]:
    """
    Divide the answer sheet into different question boxes
    """
    # convert to greyscale
    imagec = image.copy()
    gray = cv2.cvtColor(imagec, cv2.COLOR_BGR2GRAY)

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
        minDist=3 * BUBBLE_RADIUS_PX,  # min distance between centers, avoids detecting C as bubble
        param1=50,
        param2=10,
        minRadius=int(BUBBLE_RADIUS_PX - 2),
        maxRadius=int(BUBBLE_RADIUS_PX + 2),
    )

    bubbles = []

    if circles is not None:
        circles = np.around(circles[0]).astype(np.uint16)  # x,y,r

        ys = circles[:, 1]
        max_y = np.max(ys)
        tolerance = BUBBLE_RADIUS_PX

        # only keep circles that are in the bottom of the box (avoid false posititves with text)
        valid_circles = []
        for cx, cy, r in circles:
            if cy >= (max_y - tolerance):
                valid_circles.append((cx, cy, r))

        for cx, cy, r in valid_circles:
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

    # convert bubble position to the global coordinates
    bubbles = [(b[0] + box[0], b[1] + box[1], b[2], b[3]) for b in bubbles]

    return answers, bubbles


def find_writing_area(thresh: MatLike, box: tuple[int, int, int, int]):
    """
    find the area to fill in answer, returns the coordinates of each box to write in. 
    Return both local and global coordinates
    
    :param thresh: black and white image
    :type thresh: MatLike
    :param box: question box (x, y, w, h)
    :type box: tuple[int, int, int, int]
    """
    x, y, w, h = box
    roi = thresh[y: y+h, x:x+w].copy() # copy question area
    blur = cv2.GaussianBlur(roi, (3,3), 0) # blur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # dilation to merge handwriting noise
    dil = cv2.dilate(blur, kernel, iterations = 1)
    contours,  _ = cv2.findContours(
        dil,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = [] # coordinates relative to the question box
    global_results = [] # global coordinates
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if (cw > 0.90 * w and ch > 0.90 * h) or (cw < 50) or (ch < 50): # avoid detecting the question box itself or small noises
            continue
        results.append((cx, cy, cw, ch))
        global_results.append((cx + x, cy + y, cw, ch))

    if len(global_results) > 1:
        global_results = [max(global_results, key=lambda c: c[2] * c[3])]  # there are duplicates of similar size sometimes...

    return results, global_results


def numeric_box(thresh: MatLike, box: tuple[int, int, int, int], processor, model):
    """
    The complete pipeline for numeric question detection
    """
    _, global_coords = find_writing_area(thresh, box) # get the small writing boxes
    global_coords = sorted(global_coords, key=lambda c: c[0]) # make sure left most digit is processed
    digits = []

    for coord in global_coords:
        x, y, w, h = coord
        roi = thresh[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        img_rgb = Image.fromarray(roi_rgb)

        pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values 
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        digits.append(generated_text)

    return digits, global_coords
