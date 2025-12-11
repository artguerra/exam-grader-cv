import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from generator import FIDUCIAL_RADIUS
from util import mm_to_px


def detect_circles_in_page(img: MatLike, mask: npt.ArrayLike, dpi: int) -> npt.NDArray[np.int32]:
    """
    Run HoughCircles constrained to the page area.
    Returns np.ndarray of shape (N, 3) for (x, y, r) as int.
    Raises if nothing is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zero out everything outside the page to prevent weird detections
    gray_roi = gray.copy()
    gray_roi[mask == 0] = 0

    gray_roi = cv2.medianBlur(gray_roi, 5)
    fiducial_radius_px = mm_to_px(FIDUCIAL_RADIUS, dpi)

    circles = cv2.HoughCircles(
        gray_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=120,
        param2=25,
        minRadius=fiducial_radius_px - 5,
        maxRadius=fiducial_radius_px + 5,
    )

    if circles is None:
        raise RuntimeError("No circles detected inside page ROI.")
    return circles[0].astype(np.int32)  # (N,3) float


def pick_outermost_circles(
    circles: npt.NDArray[np.int32], centroid: tuple[float, float]
) -> list[tuple[int, int, int]]:
    """
    Splits the page into 4 quadrants relative to the centroid and selects the
    furthest circle in each quadrant.

    Returns list of 4 (x,y,r) ints.
    """
    cx, cy = centroid

    # for the quadrants, each item is tuple (distance_sq, circle)
    tl, tr, bl, br = [], [], [], []

    for c in circles:
        x, y, r = c
        # squared distance from center
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        
        # assign to quadrant
        if x < cx and y < cy:
            tl.append((dist_sq, c))
        elif x >= cx and y < cy:
            tr.append((dist_sq, c))
        elif x < cx and y >= cy:
            bl.append((dist_sq, c))
        elif x >= cx and y >= cy:
            br.append((dist_sq, c))

    def get_best_circle(candidates):
        if not candidates:
            # if a quadrant is empty, we cant find that corner.
            raise RuntimeError("Could not find a fiducial marker in one of the page quadrants.")

        # circle with the maximum distance
        return max(candidates, key=lambda item: item[0])[1]

    final_corners = [
        get_best_circle(tl),
        get_best_circle(tr),
        get_best_circle(bl),
        get_best_circle(br)
    ]

    return [(c[0], c[1], c[2]) for c in final_corners]


def debug_draw_circles(
    img: MatLike, circles: list[tuple[int, int, int]] | npt.NDArray[np.int32]
) -> None:
    out = img.copy()
    for x, y, r in circles:
        cv2.circle(out, (x, y), r, (255, 100, 255), 2)
        cv2.circle(out, (x, y), 2, (0, 200, 0), 3)

    cv2.imshow("image w/ circles", cv2.resize(out, (545, 842)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rectify_page(
    img: MatLike, page_mask: MatLike, mask_centroid: tuple[float, float], dpi: int
) -> MatLike:
    # detect circles
    circles = detect_circles_in_page(img, page_mask, dpi)

    # pick the corner circles (the outermost ones in the page)
    corners = pick_outermost_circles(circles, mask_centroid)
    new_corners = np.array([(c[0], c[1]) for c in corners], dtype="float32")

    s = new_corners.sum(axis=1)
    tl = new_corners[np.argmin(s)]
    br = new_corners[np.argmax(s)]
    d = np.diff(new_corners, axis=1)  # shape (4,1) => (y-x)
    tr = new_corners[np.argmin(d)]
    bl = new_corners[np.argmax(d)]

    src_pts = np.array([tl, tr, bl, br], dtype="float32") # source points

    # output size
    width = 2362
    height = 3391

    dst_pts = np.array([
        [0,0],
        [width-1, 0],
        [0, height-1],
        [width-1, height-1]
    ], dtype="float32")

    # transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

