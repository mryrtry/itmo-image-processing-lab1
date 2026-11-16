import numpy as np

KX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

KY = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]], dtype=np.float32)


def to_gray(img):
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def conv3x3(img, k):
    h, w = img.shape
    p = np.pad(img, 1, mode="constant")
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            c = k[i, j]
            if c:
                out += c * p[i:i + h, j:j + w]
    return out


def sobel_native(img):
    gray = to_gray(img).astype(np.float32)
    gx = conv3x3(gray, KX)
    gy = conv3x3(gray, KY)
    mag = np.sqrt(gx * gx + gy * gy)
    return np.clip(mag, 0, 255).astype(np.uint8)
