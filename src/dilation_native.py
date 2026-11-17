import numpy as np


def dilate_native(img):
    padded = np.pad(img, 1, mode="constant", constant_values=0)
    h, w = img.shape
    out = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            np.maximum(out, padded[i:i + h, j:j + w], out=out)
    return out
