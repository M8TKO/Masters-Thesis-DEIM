from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "image.jpg"
OUTPUT_PATH = ROOT / "svd_image_compression.png"
SINGULAR_VALUES_OUTPUT_PATH = ROOT / "svd_image_singular_values.png"

K_VALUES = [5, 20, 50, 100]
MAX_SIDE = 900


def truncated_svd_reconstruction(u, singular_values, vt, k):
    return (u[:, :k] * singular_values[:k]) @ vt[:k, :]


def storage_title(label, entries, original_entries):
    percentage = 100 * entries / original_entries
    return f"{label}\n{entries:,} values ({percentage:.1f}%)"


def main():
    image = Image.open(IMAGE_PATH)
    image = ImageOps.grayscale(image)
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    matrix = np.asarray(image, dtype=float) / 255.0
    m, n = matrix.shape
    original_entries = m * n
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    fig, axes = plt.subplots(1, len(K_VALUES) + 1, figsize=(14, 4.6), constrained_layout=True)

    axes[0].imshow(matrix, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(storage_title(f"Original {m} x {n}", original_entries, original_entries), fontsize=11)
    axes[0].axis("off")

    for axis, k in zip(axes[1:], K_VALUES):
        approximation = truncated_svd_reconstruction(u, singular_values, vt, k)
        approximation = np.clip(approximation, 0.0, 1.0)
        approximation_entries = k * (m + n + 1)

        axis.imshow(approximation, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(storage_title(f"$k={k}$", approximation_entries, original_entries), fontsize=11)
        axis.axis("off")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    indices = np.arange(1, len(singular_values) + 1)
    normalized_singular_values = singular_values / singular_values[0]

    fig, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    axis.semilogy(indices, normalized_singular_values, color="#1f77b4", linewidth=1.8)
    for k in K_VALUES:
        axis.axvline(k, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.7)
    axis.set_xlabel("Index $i$")
    axis.set_ylabel(r"$\sigma_i / \sigma_1$")
    axis.set_title("Singular values decay")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    fig.savefig(SINGULAR_VALUES_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
