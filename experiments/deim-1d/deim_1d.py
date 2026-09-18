import numpy as np

from plotting import plot_deim_approximation, plot_snapshots


def exact_function(x, p):
    return (x-0.5)**2 * np.sin(np.exp(11.2*np.abs(p - 0.5)) * x * (1-x)) 


def create_grid(number_of_points=400):
    return np.linspace(0, 1, number_of_points)


def snapshot_matrix(x, parameters):
    return np.column_stack([exact_function(x, parameter) for parameter in parameters])


def pod_basis(snapshots, number_of_basis_vectors):
    left_singular_vectors, _, _ = np.linalg.svd(snapshots, full_matrices=False)
    return left_singular_vectors[:, :number_of_basis_vectors]


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


def deim_approximation(values, basis, indices):
    coefficients = np.linalg.solve(basis[indices, :], values[indices])
    return basis @ coefficients


def run_demo():
    x = create_grid()
    training_parameters = np.linspace(0.05, 1.0, 40)
    plotted_parameters = [0.1, 0.4, 0.5, 0.7, 1.0]
    number_of_basis_vectors = 30

    snapshots = snapshot_matrix(x, training_parameters)
    basis = pod_basis(snapshots, number_of_basis_vectors)
    indices = deim_indices(basis)

    plot_snapshots(
        x,
        exact_function,
        plotted_parameters,
        deim_indices=None,
        save_name="exact_function_1d_snapshots.png",
        show=False,
    )

    test_parameters = [0.12, 0.28, 0.44, 0.63, 0.82, 0.96]
    exact_values_list = [exact_function(x, parameter) for parameter in test_parameters]
    approximation_values_list = [
        deim_approximation(exact_values, basis, indices) for exact_values in exact_values_list
    ]

    plot_deim_approximation(
        x,
        exact_values_list,
        approximation_values_list,
        indices,
        parameters=test_parameters,
        save_name="deim_1d_approximation.png",
        show=False,
    )

    print("DEIM indices:", indices)
    print("DEIM points:", x[indices])


f = exact_function


if __name__ == "__main__":
    run_demo()
