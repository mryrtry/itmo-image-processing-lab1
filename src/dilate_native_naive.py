import numpy as np


def dilate_native_naive(img):
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            m = 0
            y0 = y - 1 if y > 0 else 0
            y1 = y + 2 if y < h - 1 else h
            x0 = x - 1 if x > 0 else 0
            x1 = x + 2 if x < w - 1 else w
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    v = img[yy, xx]
                    if v > m:
                        m = v
            out[y, x] = m
    return out
