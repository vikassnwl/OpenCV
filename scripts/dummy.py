import cv2  # type: ignore
import numpy as np
from typing import Union

Mat = np.ndarray

canvas: Mat = np.array([[0]], np.float64)
img: Mat = np.uint8(canvas)
print(img)