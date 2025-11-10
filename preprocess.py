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

    # Get max area contour
    c = max(contours, key=cv2.contourArea)

    # Approximate to polygon; try to get 4 vertices
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)

    # If we didn't get 4 points, just use its min-area rectangle as a fallback
    if len(approx) != 4:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)  # 4x2 float
        quad = box.astype(np.int32)
    else:
        quad = approx.reshape(-1, 2).astype(np.int32)

    mask = np.zeros_like(thresh)
    cv2.fillPoly(mask, [quad], 255)

    # cv2.imshow("image with page mask applied", cv2.resize(cv2.bitwise_and(image, image, mask=mask), (545, 842)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask, quad
