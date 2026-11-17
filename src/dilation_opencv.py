import cv2
import numpy as np


def dilate_opencv(img):
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(img, kernel, iterations=1)
