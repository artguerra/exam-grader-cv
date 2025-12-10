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
    circles: npt.NDArray[np.int32], page_bbox: npt.NDArray[np.int32]
) -> list[tuple[int, int, int]]:
    """
    Given all circles (x,y,r) and the page bounding box, pick the 4 furthest from
    the page center.
    Returns list of 4 (x,y,r) ints.
    """
    cx, cy = page_bbox[:, 0].mean(), page_bbox[:, 1].mean()

    # Sort by distance from page center, descending
    dists = [((c[0] - cx) ** 2 + (c[1] - cy) ** 2, i) for i, c in enumerate(circles)]
    dists.sort(reverse=True)

    top4 = [circles[i] for (_, i) in dists[:4]]
    return [(c[0], c[1], c[2]) for c in top4]


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
    img: MatLike, page_mask: MatLike, page_bbox: npt.NDArray[np.int32], dpi: int
) -> MatLike:
    #  Detect circles
    circles = detect_circles_in_page(img, page_mask, dpi)

    # Pick the corner circles (the outermost ones in the page)
    corners = pick_outermost_circles(circles, page_bbox)
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

    


