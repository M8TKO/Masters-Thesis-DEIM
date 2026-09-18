from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.linalg import qr


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "image.jpg"
OUTPUT_PATH = ROOT / "qr_image_compression.png"

K_VALUES = [5, 20, 50, 100]
MAX_SIDE = 900


def qr_reconstruction(q, matrix, k):
    q_k = q[:, :k]
    return q_k @ (q_k.T @ matrix)


def main():
    image = Image.open(IMAGE_PATH)
    image = ImageOps.grayscale(image)
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    matrix = np.asarray(image, dtype=float) / 255.0
    q, _, _ = qr(matrix, pivoting=True, mode="economic")

    fig, axes = plt.subplots(1, len(K_VALUES) + 1, figsize=(14, 4.6), constrained_layout=True)

    axes[0].imshow(matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for axis, k in zip(axes[1:], K_VALUES):
        approximation = qr_reconstruction(q, matrix, k)
        approximation = np.clip(approximation, 0.0, 1.0)

        axis.imshow(approximation, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"$k={k}$", fontsize=11)
        axis.axis("off")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
