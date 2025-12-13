import cv2
import numpy as np
from cv2.typing import MatLike


def detect_page_mask(img: MatLike) -> tuple[MatLike, tuple[float, float]]:
    """
    Detects the page using Canny Edge Detection, robust to complex backgrounds
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # use canny edge detection instead of thresholding
    # these values (30, 150) seem to work well for fairly high contrast document edges
    edges = cv2.Canny(blur, 30, 150)

    # dilate the edges to close small gaps
    # tries to make the outline of the region a continuous loop
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # find contours on the edge map
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise RuntimeError("No external contour found for page.")

    h, w = img.shape[:2]
    img_area = w * h
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)

    # top 10 contours by perimeter
    candidates = contours[:10]

    # small contours can make a big diff in the convex hull way bigger, so filter
    valid_contours = []
    min_area_threshold = 0.10 * img_area

    for c in candidates:
        # check bounding rect area because lines have low contourArea
        _, _, cw, ch = cv2.boundingRect(c)
        if (cw * ch) > min_area_threshold:
            valid_contours.append(c)

    # if everything was filtered out, fallback to the biggest perimeter
    if not valid_contours:
        valid_contours = [contours[0]]

    # combine points from all valid edge pieces
    # and do a convex hull of the union
    all_points = np.vstack(valid_contours)
    hull = cv2.convexHull(all_points)
    
    # if hull is still too small, use full image
    if cv2.contourArea(hull) < 0.5 * img_area:
        print("Warning: Detected area too small, using full image.")
        hull = np.array([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]], dtype=np.int32)

    # create the mask
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [hull], 255)

    points = hull.reshape(-1, 2)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    cx = int((min_x + max_x) / 2)
    cy = int((min_y + max_y) / 2)

    # vis = img.copy()
    # cv2.drawContours(vis, [hull], 0, (0, 255, 0), 2)
    # cv2.circle(vis, (cx, cy), 24, (0, 0, 255), -1)
    # cv2.imwrite("hull.jpg", vis)

    return mask, (cx, cy)
