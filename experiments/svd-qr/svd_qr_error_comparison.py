from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.linalg import qr


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "image.jpg"
OUTPUT_PATH = ROOT / "svd_qr_error_comparison.png"

MAX_SIDE = 900
MAX_K = 100


def main():
    image = Image.open(IMAGE_PATH)
    image = ImageOps.grayscale(image)
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    matrix = np.asarray(image, dtype=float) / 255.0
    matrix_norm = np.linalg.norm(matrix, ord="fro")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    _, r, _ = qr(matrix, pivoting=True, mode="economic")

    k_values = np.arange(1, min(MAX_K, len(singular_values)) + 1)

    squared_singular_values = singular_values**2
    svd_tail_energy = np.cumsum(squared_singular_values[::-1])[::-1]
    svd_errors = np.sqrt(svd_tail_energy[k_values]) / matrix_norm

    squared_r_row_norms = np.sum(r**2, axis=1)
    qr_tail_energy = np.cumsum(squared_r_row_norms[::-1])[::-1]
    qr_errors = np.sqrt(qr_tail_energy[k_values]) / matrix_norm

    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.semilogy(k_values, svd_errors, label="Truncated SVD", linewidth=2.0)
    axis.semilogy(k_values, qr_errors, label="Column-pivoted QR", linewidth=2.0)
    axis.set_xlabel("Number of basis vectors $k$")
    axis.set_ylabel(r"Relative Frobenius error")
    axis.set_title("SVD and pivoted QR reconstruction error")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    axis.legend()

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
