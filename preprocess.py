import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

def detect_page_mask(img: MatLike) -> tuple[MatLike, npt.NDArray[np.int32]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No external contour found for page.")

    # get max area contour
    h, w = img.shape[:2]
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # approximate to polygon; try to get 4 vertices
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)

    # if not 4 points or too small to be the page, fall back to full image
    use_full_image = False

    if len(approx) != 4 or area < 0.5 * (w * h):
        use_full_image = True

    if use_full_image:
        quad = np.array(
            [
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0],
            ],
            dtype=np.int32,
        )
    else:
        quad = approx.reshape(-1, 2).astype(np.int32)

    mask = np.zeros_like(thresh)
    cv2.fillPoly(mask, [quad], 255)

    # vis = img.copy()
    # cv2.imshow(
    #     "image with page mask applied",
    #     cv2.resize(cv2.bitwise_and(vis, vis, mask=mask), (545, 842)),
    # )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask, quad
