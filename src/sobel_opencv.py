import cv2
import numpy as np


def sobel_opencv(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_CONSTANT)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)
