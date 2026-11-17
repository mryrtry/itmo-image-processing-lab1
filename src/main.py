import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dilation_native import dilate_native
from dilation_opencv import dilate_opencv
from benchmark import benchmark

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES_DIR = RESULTS / "images"


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def quality(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "shape": a.shape,
        "max_diff": int(d.max()),
        "mean_diff": float(d.mean()),
        "equal_pct": float((d == 0).mean() * 100.0),
    }


def save_comparison(orig, bw, native, opencv, out):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Оригинал", fontsize=13, pad=10)
    ax[1].imshow(bw, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Бинаризация (Otsu)", fontsize=13, pad=10)
    ax[2].imshow(native, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("Дилатация — native (NumPy)", fontsize=13, pad=10)
    ax[3].imshow(opencv, cmap="gray", vmin=0, vmax=255)
    ax[3].set_title("Дилатация — OpenCV", fontsize=13, pad=10)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def run_bench(bw, sizes, out_csv):
    rows = []
    for s in sizes:
        im = cv2.resize(bw, (s, s), interpolation=cv2.INTER_NEAREST)
        tn, sn, nn = benchmark(dilate_native, im)
        tc, sc, nc = benchmark(dilate_opencv, im)
        sp = tn / tc if tc else float("inf")
        rows.append({
            "size": f"{s}x{s}",
            "pixels": s * s,
            "native_mean_ms": tn * 1000,
            "native_std_ms": sn * 1000,
            "native_iters": nn,
            "opencv_mean_ms": tc * 1000,
            "opencv_std_ms": sc * 1000,
            "opencv_iters": nc,
            "speedup_x": sp,
        })
        print(f"[{s}x{s}] native={tn * 1000:8.2f} ms  opencv={tc * 1000:6.2f} ms  speedup={sp:6.1f}x")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def plot_bench(df, out):
    labels = df["size"].tolist()
    x = np.arange(len(labels))
    w = 0.38
    nat = df["native_mean_ms"].to_numpy()
    ocl = df["opencv_mean_ms"].to_numpy()
    spd = df["speedup_x"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x - w / 2, nat, width=w, label="native (NumPy)", color="#4C72B0")
    ax.bar(x + w / 2, ocl, width=w, label="OpenCV", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Размер изображения")
    ax.set_ylabel("Время на один вызов, мс")
    ax.set_title("Дилатация 3×3: native vs OpenCV")
    ax.grid(True, axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    for i, (n, o, s) in enumerate(zip(nat, ocl, spd)):
        ax.text(x[i], max(n, o) * 1.02, f"×{s:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=str(ROOT / "data" / "input" / "test_image.png"))
    p.add_argument("--no-bench", action="store_true")
    args = p.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    img = load_image(Path(args.image))
    print(f"image: {args.image}")
    print(f"shape: {img.shape}, dtype: {img.dtype}")

    bw = binarize(img)
    print(f"binary: shape={bw.shape}, unique={np.unique(bw).tolist()}")

    nat = dilate_native(bw)
    ocl = dilate_opencv(bw)

    q = quality(nat, ocl)
    print("native vs opencv (dilate):")
    for k, v in q.items():
        print(f"  {k}: {v}")

    save_comparison(img, bw, nat, ocl, IMAGES_DIR / "comparison.png")
    cv2.imwrite(str(IMAGES_DIR / "binary.png"), bw)
    cv2.imwrite(str(IMAGES_DIR / "native.png"), nat)
    cv2.imwrite(str(IMAGES_DIR / "opencv.png"), ocl)
    print(f"saved: {IMAGES_DIR / 'comparison.png'}")

    if args.no_bench:
        return

    sizes = [256, 512, 1024, 2048]
    print("\nbenchmark:")
    df = run_bench(bw, sizes, RESULTS / "benchmark.csv")
    plot_bench(df, IMAGES_DIR / "benchmark.png")
    print(f"csv: {RESULTS / 'benchmark.csv'}")
    print(f"plot: {IMAGES_DIR / 'benchmark.png'}")


if __name__ == "__main__":
    main()
