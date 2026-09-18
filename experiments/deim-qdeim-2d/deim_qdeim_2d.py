from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr


FILE_PATH = Path(__file__).resolve()
PLOTS_DIR = FILE_PATH.parent / "plots"


def exact_function(x, y, mu1, mu2):
    return np.exp(-40.0 * ((x - mu1) ** 2 + (y - mu2) ** 2))


def create_spatial_grid(number_of_points=80):
    grid = np.linspace(0.0, 1.0, number_of_points)
    return np.meshgrid(grid, grid, indexing="xy")


def create_parameter_grid(number_of_points=15):
    values = np.linspace(0.2, 0.8, number_of_points)
    return [(mu1, mu2) for mu1 in values for mu2 in values]


def snapshot_matrix(x_grid, y_grid, parameters):
    snapshots = [
        exact_function(x_grid, y_grid, mu1, mu2).reshape(-1)
        for mu1, mu2 in parameters
    ]
    return np.column_stack(snapshots)


def pod_basis(snapshots, number_of_basis_vectors):
    left_singular_vectors, singular_values, _ = np.linalg.svd(snapshots, full_matrices=False)
    return left_singular_vectors[:, :number_of_basis_vectors], singular_values


def deim_indices(basis):
    first_index = int(np.argmax(np.abs(basis[:, 0])))
    indices = [first_index]

    for ell in range(1, basis.shape[1]):
        current_basis = basis[:, :ell]
        current_vector = basis[:, ell]

        coefficients = np.linalg.solve(
            current_basis[indices, :],
            current_vector[indices],
        )
        residual = current_vector - current_basis @ coefficients
        indices.append(int(np.argmax(np.abs(residual))))

    return np.array(indices, dtype=int)


def qdeim_indices(basis):
    _, _, pivots = qr(basis.T, pivoting=True)
    return np.array(pivots[: basis.shape[1]], dtype=int)


def deim_approximation(values, basis, indices):
    coefficients = np.linalg.solve(basis[indices, :], values[indices])
    return basis @ coefficients


def relative_error(exact_values, approximation_values):
    return np.linalg.norm(exact_values - approximation_values) / np.linalg.norm(exact_values)


def unravel_indices(indices, grid_shape):
    row_indices, col_indices = np.unravel_index(indices, grid_shape)
    return row_indices, col_indices


def save_figure(fig, save_name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / save_name, dpi=300, bbox_inches="tight")


def plot_selected_indices(x_grid, y_grid, deim_points, qdeim_points, save_name):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    deim_rows, deim_cols = unravel_indices(deim_points, x_grid.shape)
    qdeim_rows, qdeim_cols = unravel_indices(qdeim_points, x_grid.shape)

    ax.scatter(
        x_grid[deim_rows, deim_cols],
        y_grid[deim_rows, deim_cols],
        s=38,
        color="tab:red",
        label="DEIM",
        alpha=0.9,
    )
    ax.scatter(
        x_grid[qdeim_rows, qdeim_cols],
        y_grid[qdeim_rows, qdeim_cols],
        s=46,
        marker="x",
        color="tab:blue",
        label="QDEIM",
        linewidths=1.8,
    )

    ax.set_title("Selected interpolation points")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def plot_reconstruction(x_grid, y_grid, exact_values, deim_values, qdeim_values, save_name):
    exact_image = exact_values.reshape(x_grid.shape)
    deim_image = deim_values.reshape(x_grid.shape)
    qdeim_image = qdeim_values.reshape(x_grid.shape)

    fig = plt.figure(figsize=(12.0, 4.2), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 3, index, projection="3d")
        for index in range(1, 4)
    ]
    images = [
        (exact_image, "Exact"),
        (deim_image, "DEIM"),
        (qdeim_image, "QDEIM"),
    ]

    for ax, (image, title) in zip(axes, images):
        ax.plot_surface(
            x_grid,
            y_grid,
            image,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$f$")
        ax.set_zlim(-0.05, 1.05)
        ax.view_init(elev=28, azim=-55)

    save_figure(fig, save_name)
    plt.close(fig)


def plot_error_curve(m_values, deim_errors, qdeim_errors, save_name):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(m_values, deim_errors, "o-", color="tab:red", label="DEIM")
    ax.semilogy(m_values, qdeim_errors, "x-", color="tab:blue", label="QDEIM")
    ax.set_title("Mean relative reconstruction error")
    ax.set_xlabel(r"number of basis vectors $m$")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def run_demo():
    x_grid, y_grid = create_spatial_grid(number_of_points=80)
    training_parameters = create_parameter_grid(number_of_points=15)
    test_parameters = [(0.25, 0.35), (0.37, 0.71), (0.52, 0.48), (0.74, 0.62)]
    visualization_basis_vectors = 40
    max_error_basis_vectors = 100

    snapshots = snapshot_matrix(x_grid, y_grid, training_parameters)
    basis, singular_values = pod_basis(snapshots, max_error_basis_vectors)

    visualization_basis = basis[:, :visualization_basis_vectors]
    deim_points = deim_indices(visualization_basis)
    qdeim_points = qdeim_indices(visualization_basis)

    plot_selected_indices(
        x_grid,
        y_grid,
        deim_points,
        qdeim_points,
        save_name="deim_qdeim_2d_selected_points.png",
    )

    representative_parameter = (0.63, 0.41)
    exact_values = exact_function(x_grid, y_grid, *representative_parameter).reshape(-1)
    deim_values = deim_approximation(exact_values, visualization_basis, deim_points)
    qdeim_values = deim_approximation(exact_values, visualization_basis, qdeim_points)
    plot_reconstruction(
        x_grid,
        y_grid,
        exact_values,
        deim_values,
        qdeim_values,
        save_name="deim_qdeim_2d_reconstruction.png",
    )

    m_values = np.arange(5, max_error_basis_vectors + 1)
    deim_errors = []
    qdeim_errors = []

    for m in m_values:
        basis_m = basis[:, :m]
        deim_points_m = deim_indices(basis_m)
        qdeim_points_m = qdeim_indices(basis_m)

        errors_deim_m = []
        errors_qdeim_m = []
        for mu1, mu2 in test_parameters:
            exact_m = exact_function(x_grid, y_grid, mu1, mu2).reshape(-1)
            deim_m = deim_approximation(exact_m, basis_m, deim_points_m)
            qdeim_m = deim_approximation(exact_m, basis_m, qdeim_points_m)
            errors_deim_m.append(relative_error(exact_m, deim_m))
            errors_qdeim_m.append(relative_error(exact_m, qdeim_m))

        deim_errors.append(np.mean(errors_deim_m))
        qdeim_errors.append(np.mean(errors_qdeim_m))

    plot_error_curve(
        m_values,
        deim_errors,
        qdeim_errors,
        save_name="deim_qdeim_2d_error_curve.png",
    )

    print("Spatial grid:", x_grid.shape[0], "x", x_grid.shape[1])
    print("Training parameters:", len(training_parameters))
    print("Visualization basis vectors:", visualization_basis_vectors)
    print("Error curve basis vectors:", f"{m_values[0]} to {m_values[-1]}")
    print("First DEIM indices:", deim_points[:10])
    print("First QDEIM indices:", qdeim_points[:10])
    print("Singular values, first five:", singular_values[:5])
    print("Mean DEIM errors:", np.array(deim_errors))
    print("Mean QDEIM errors:", np.array(qdeim_errors))


f = exact_function


if __name__ == "__main__":
    run_demo()
