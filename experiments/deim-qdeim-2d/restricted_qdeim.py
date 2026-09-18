from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.linalg import qr

from deim_qdeim_2d import (
    create_parameter_grid,
    create_spatial_grid,
    deim_approximation,
    exact_function,
    pod_basis,
    relative_error,
    snapshot_matrix,
)


FILE_PATH = Path(__file__).resolve()
PLOTS_DIR = FILE_PATH.parent / "plots"


def qdeim_indices(basis):
    _, _, pivots = qr(basis.T, pivoting=True)
    return np.array(pivots[: basis.shape[1]], dtype=int)


def restricted_qdeim_indices(basis, admissible_indices):
    admissible_basis = basis[admissible_indices, :]
    _, _, local_pivots = qr(admissible_basis.T, pivoting=True)
    return admissible_indices[local_pivots[: basis.shape[1]]]


def save_figure(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")


def coordinates_from_indices(x_grid, y_grid, indices):
    return x_grid.reshape(-1)[indices], y_grid.reshape(-1)[indices]


def plot_selected_points(x_grid, y_grid, unrestricted, restricted):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True, sharey=True)

    for ax, indices, title in [
        (axes[0], unrestricted, "Unrestricted QDEIM"),
        (axes[1], restricted, "Restricted QDEIM"),
    ]:
        x_points, y_points = coordinates_from_indices(x_grid, y_grid, indices)
        ax.add_patch(
            Rectangle(
                (0.35, 0.35),
                0.30,
                0.30,
                facecolor="lightgray",
                edgecolor="black",
                hatch="//",
                alpha=0.7,
                label="Inaccessible region",
            )
        )
        ax.scatter(x_points, y_points, s=25, color="tab:red", zorder=3)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$y$")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "restricted_qdeim_2d_points.png")
    plt.close(fig)


def plot_errors_and_constants(
    dimensions,
    unrestricted_errors,
    restricted_errors,
    unrestricted_constants,
    restricted_constants,
):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    axes[0].semilogy(
        dimensions,
        unrestricted_errors,
        "o-",
        label="Unrestricted QDEIM",
    )
    axes[0].semilogy(
        dimensions,
        restricted_errors,
        "s--",
        label="Restricted QDEIM",
    )
    axes[0].set_xlabel(r"basis dimension $m$")
    axes[0].set_ylabel("mean relative error")
    axes[0].set_title("Reconstruction error")

    axes[1].semilogy(
        dimensions,
        unrestricted_constants,
        "o-",
        label="Unrestricted QDEIM",
    )
    axes[1].semilogy(
        dimensions,
        restricted_constants,
        "s--",
        label="Restricted QDEIM",
    )
    axes[1].set_xlabel(r"basis dimension $m$")
    axes[1].set_ylabel(r"$\|(\mathbf{P}^T\mathbf{U}_m)^{-1}\|_2$")
    axes[1].set_title("Interpolation constant")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    save_figure(fig, "restricted_qdeim_2d_error_conditioning.png")
    plt.close(fig)


def run_experiment():
    x_grid, y_grid = create_spatial_grid(number_of_points=80)
    training_parameters = create_parameter_grid(number_of_points=15)
    test_parameters = [
        (0.25, 0.35),
        (0.37, 0.71),
        (0.52, 0.48),
        (0.74, 0.62),
    ]
    dimensions = np.arange(5, 81, 5)

    snapshots = snapshot_matrix(x_grid, y_grid, training_parameters)
    basis, _ = pod_basis(snapshots, number_of_basis_vectors=dimensions[-1])

    inaccessible = (
        (x_grid >= 0.35)
        & (x_grid <= 0.65)
        & (y_grid >= 0.35)
        & (y_grid <= 0.65)
    )
    admissible_indices = np.flatnonzero(~inaccessible.reshape(-1))

    unrestricted_errors = []
    restricted_errors = []
    unrestricted_constants = []
    restricted_constants = []

    for dimension in dimensions:
        basis_m = basis[:, :dimension]
        unrestricted = qdeim_indices(basis_m)
        restricted = restricted_qdeim_indices(basis_m, admissible_indices)

        errors_unrestricted = []
        errors_restricted = []
        for mu1, mu2 in test_parameters:
            exact_values = exact_function(
                x_grid,
                y_grid,
                mu1,
                mu2,
            ).reshape(-1)
            approximation_unrestricted = deim_approximation(
                exact_values,
                basis_m,
                unrestricted,
            )
            approximation_restricted = deim_approximation(
                exact_values,
                basis_m,
                restricted,
            )
            errors_unrestricted.append(
                relative_error(exact_values, approximation_unrestricted)
            )
            errors_restricted.append(
                relative_error(exact_values, approximation_restricted)
            )

        unrestricted_errors.append(np.mean(errors_unrestricted))
        restricted_errors.append(np.mean(errors_restricted))
        unrestricted_constants.append(
            np.linalg.norm(np.linalg.inv(basis_m[unrestricted, :]), 2)
        )
        restricted_constants.append(
            np.linalg.norm(np.linalg.inv(basis_m[restricted, :]), 2)
        )

        if dimension == 40:
            plot_selected_points(
                x_grid,
                y_grid,
                unrestricted,
                restricted,
            )
            print("Dimension 40")
            print("Unrestricted mean error:", unrestricted_errors[-1])
            print("Restricted mean error:", restricted_errors[-1])
            print("Unrestricted interpolation constant:", unrestricted_constants[-1])
            print("Restricted interpolation constant:", restricted_constants[-1])
            print(
                "Rank of admissible basis:",
                np.linalg.matrix_rank(basis_m[admissible_indices, :]),
            )

    plot_errors_and_constants(
        dimensions,
        unrestricted_errors,
        restricted_errors,
        unrestricted_constants,
        restricted_constants,
    )


if __name__ == "__main__":
    run_experiment()
