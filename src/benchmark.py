import time

import numpy as np


def benchmark(fn, img, warmup=3, min_time=0.5, max_iter=200):
    for _ in range(warmup):
        fn(img)

    times = []
    total = 0.0
    while total < min_time and len(times) < max_iter:
        t0 = time.perf_counter()
        fn(img)
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt

    return float(np.mean(times)), float(np.std(times)), len(times)
