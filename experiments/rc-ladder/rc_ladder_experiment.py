"""POD, DEIM, QDEIM, and SNS comparison for a nonlinear RC ladder."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr
from scipy.sparse import diags


def diode_resistor_current(voltage):
    """Current law g(z) = exp(40 z) + z - 1."""
    return np.exp(40.0 * voltage) + voltage - 1.0


def input_current(t):
    return np.exp(-t)


def internal_rhs(state):
    """Evaluate -C.T g(C state) without explicitly forming C."""
    edge_voltage = np.empty_like(state)
    edge_voltage[0] = state[0]
    edge_voltage[1:] = state[:-1] - state[1:]

    current = diode_resistor_current(edge_voltage)
    value = np.empty_like(state)
    value[0] = -current[0] - current[1]
    value[1:-1] = current[1:-1] - current[2:]
    value[-1] = current[-1]
    return value


def full_rhs(t, state):
    value = internal_rhs(state)
    value[0] += input_current(t)
    return value


def solve_model(rhs, initial_state, final_time, output_times, jacobian_sparsity=None):
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_state,
        t_eval=output_times,
        method="BDF",
        rtol=1.0e-8,
        atol=1.0e-10,
        jac_sparsity=jacobian_sparsity,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def svd_basis(snapshots):
    basis, singular_values, _ = np.linalg.svd(snapshots, full_matrices=False)
    return basis, singular_values


def deim_indices(basis):
    """Original greedy DEIM index selection."""
    indices = [int(np.argmax(np.abs(basis[:, 0])))]

    for column in range(1, basis.shape[1]):
        previous_basis = basis[:, :column]
        sampled_basis = previous_basis[indices, :]
        coefficients = np.linalg.solve(
            sampled_basis,
            basis[indices, column],
        )
        residual = basis[:, column] - previous_basis @ coefficients
        indices.append(int(np.argmax(np.abs(residual))))

    return np.array(indices, dtype=int)


def qdeim_indices(basis):
    """QDEIM selection by column-pivoted QR of the transposed basis."""
    _, _, pivots = qr(basis.T, pivoting=True, mode="economic")
    return np.asarray(pivots[: basis.shape[1]], dtype=int)


def sampling_matrices(indices, state_basis):
    """
    Build the exact local evaluator P.T F(V a) = L g(H a).

    Edge 0 carries voltage v_0. Edge j > 0 carries v_{j-1} - v_j.
    A selected node needs only its incident edges.
    """
    n = state_basis.shape[0]
    required_edges = set()

    for node in indices:
        if node == 0:
            required_edges.add(0)
        if node > 0:
            required_edges.add(int(node))
        if node < n - 1:
            required_edges.add(int(node + 1))

    edges = np.array(sorted(required_edges), dtype=int)
    edge_position = {edge: position for position, edge in enumerate(edges)}

    H = np.empty((len(edges), state_basis.shape[1]))
    for row, edge in enumerate(edges):
        if edge == 0:
            H[row, :] = state_basis[0, :]
        else:
            H[row, :] = state_basis[edge - 1, :] - state_basis[edge, :]

    L = np.zeros((len(indices), len(edges)))
    for row, node in enumerate(indices):
        if node == 0:
            L[row, edge_position[0]] -= 1.0
        if node > 0:
            L[row, edge_position[int(node)]] += 1.0
        if node < n - 1:
            L[row, edge_position[int(node + 1)]] -= 1.0

    selected_input = (indices == 0).astype(float)
    return edges, H, L, selected_input


def sampled_rhs_from_reduced(t, reduced_state, H, L, selected_input):
    sampled_value = L @ diode_resistor_current(H @ reduced_state)
    return sampled_value + selected_input * input_current(t)


def interpolation_operator(state_basis, rhs_basis, indices):
    sampled_basis = rhs_basis[indices, :]
    projected_basis = state_basis.T @ rhs_basis
    operator = np.linalg.solve(sampled_basis.T, projected_basis.T).T
    return operator, np.linalg.cond(sampled_basis)


def solve_pod(state_basis, final_time, output_times):
    reduced_initial = np.zeros(state_basis.shape[1])

    def reduced_rhs(t, reduced_state):
        state = state_basis @ reduced_state
        return state_basis.T @ full_rhs(t, state)

    return solve_model(reduced_rhs, reduced_initial, final_time, output_times)


def solve_hyper_reduced(
    state_basis,
    rhs_basis,
    selector,
    final_time,
    output_times,
):
    indices = selector(rhs_basis)
    operator, condition_number = interpolation_operator(
        state_basis,
        rhs_basis,
        indices,
    )
    edges, H, L, selected_input = sampling_matrices(indices, state_basis)
    reduced_initial = np.zeros(state_basis.shape[1])

    def reduced_rhs(t, reduced_state):
        sampled_value = sampled_rhs_from_reduced(
            t,
            reduced_state,
            H,
            L,
            selected_input,
        )
        return operator @ sampled_value

    reduced_solution = solve_model(
        reduced_rhs,
        reduced_initial,
        final_time,
        output_times,
    )
    return reduced_solution, indices, edges, condition_number


def relative_error(reference, approximation):
    return np.linalg.norm(reference - approximation) / np.linalg.norm(reference)


def validate_local_evaluator(state_basis, indices):
    edges, H, L, selected_input = sampling_matrices(indices, state_basis)
    reduced_state = np.linspace(-0.01, 0.01, state_basis.shape[1])
    t = 0.37
    state = state_basis @ reduced_state

    direct = full_rhs(t, state)[indices]
    local = sampled_rhs_from_reduced(t, reduced_state, H, L, selected_input)
    error = np.linalg.norm(direct - local) / max(
        np.linalg.norm(direct),
        np.finfo(float).eps,
    )
    return error, edges


def plot_singular_values(output_directory, state_values, rhs_values):
    fig, ax = plt.subplots(figsize=(8, 5))
    number_of_modes = min(40, len(state_values), len(rhs_values))
    mode_numbers = np.arange(1, number_of_modes + 1)
    ax.semilogy(
        mode_numbers,
        state_values[:number_of_modes],
        "o-",
        markersize=3,
        label="Solution snapshots",
    )
    ax.semilogy(
        mode_numbers,
        rhs_values[:number_of_modes],
        "s-",
        markersize=3,
        label="Right-hand-side snapshots",
    )
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value")
    ax.set_title("Nonlinear RC ladder: singular values decay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "rc_ladder_singular_values.png", dpi=180)
    plt.close(fig)


def plot_errors(output_directory, dimensions, errors):
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "POD": ("o-", "C2"),
        "DEIM": ("s:", "C3"),
        "QDEIM": ("d-.", "C4"),
        "SNS-DEIM": ("^--", "C1"),
        "SNS-QDEIM": ("v--", "C0"),
    }

    for name, values in errors.items():
        style, color = styles[name]
        ax.semilogy(dimensions, values, style, color=color, label=name)

    ax.set_xlabel("State and interpolation dimension")
    ax.set_ylabel("Relative trajectory error")
    ax.set_title("Nonlinear RC ladder: reduced-model error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "rc_ladder_error_vs_dimension.png", dpi=180)
    plt.close(fig)


def plot_node_voltage(output_directory, times, full_solution, trajectories, dimension):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    axes[0].plot(
        times,
        full_solution[0, :],
        "k-",
        linewidth=2.0,
        label="Full model",
    )

    styles = {
        "POD": ("-", "C2"),
        "DEIM": (":", "C3"),
        "QDEIM": ("-.", "C4"),
        "SNS-DEIM": ("--", "C1"),
        "SNS-QDEIM": ("--", "C0"),
    }
    for name, trajectory in trajectories.items():
        line_style, color = styles[name]
        axes[0].plot(
            times,
            trajectory[0, :],
            linestyle=line_style,
            color=color,
            linewidth=1.3,
            label=name,
        )
        axes[1].semilogy(
            times,
            np.abs(full_solution[0, :] - trajectory[0, :]) + np.finfo(float).eps,
            linestyle=line_style,
            color=color,
            linewidth=1.3,
            label=name,
        )

    axes[0].set_ylabel(r"Node-1 voltage, $v_1(t)$")
    axes[0].set_title(f"Nonlinear RC ladder: node-1 voltage, dimension {dimension}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_directory / "rc_ladder_node1_voltage.png", dpi=180)
    plt.close(fig)


def plot_sampling_stencils(output_directory, selections, n):
    fig, axes = plt.subplots(len(selections), 1, figsize=(9, 7), sharex=True)
    largest_required_node = 0

    for ax, (name, data) in zip(axes, selections.items()):
        indices = data["indices"]
        edges = data["edges"]
        required_nodes = set()
        for edge in edges:
            if edge == 0:
                required_nodes.add(0)
            else:
                required_nodes.add(int(edge - 1))
                required_nodes.add(int(edge))
        required_nodes = np.array(sorted(required_nodes), dtype=int)
        largest_required_node = max(largest_required_node, int(required_nodes[-1]))

        ax.scatter(
            required_nodes + 1,
            np.zeros(len(required_nodes)),
            marker="|",
            s=180,
            color="0.65",
            label="Required state nodes",
        )
        ax.scatter(
            indices + 1,
            np.zeros(len(indices)),
            marker="o",
            s=34,
            color="C3",
            label="Selected output nodes",
        )
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(True, axis="x", alpha=0.2)

    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlim(1, min(n, largest_required_node + 20))
    axes[-1].set_xlabel(f"Circuit node (out of {n})")
    fig.suptitle("Selected outputs and their required local state nodes")
    fig.tight_layout()
    fig.savefig(output_directory / "rc_ladder_sampling_stencils.png", dpi=180)
    plt.close(fig)


def parse_dimensions(value):
    if ":" in value:
        first, last = [int(item) for item in value.split(":")]
        return list(range(first, last + 1, 2))
    return [int(item) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--final-time", type=float, default=7.0)
    parser.add_argument("--snapshots", type=int, default=1425)
    parser.add_argument("--dimensions", default="4:20")
    parser.add_argument("--display-dimension", type=int, default=10)
    parser.add_argument(
        "--output-directory",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    args = parser.parse_args()

    dimensions = parse_dimensions(args.dimensions)
    if args.display_dimension not in dimensions:
        dimensions.append(args.display_dimension)
        dimensions.sort()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, args.final_time, args.snapshots)
    initial_state = np.zeros(args.n)
    jacobian_sparsity = diags(
        [
            np.ones(args.n - 1),
            np.ones(args.n),
            np.ones(args.n - 1),
        ],
        offsets=[-1, 0, 1],
        format="csr",
    )

    print(f"Solving the n = {args.n} nonlinear RC ladder...")
    start = time.perf_counter()
    full_solution = solve_model(
        full_rhs,
        initial_state,
        args.final_time,
        times,
        jacobian_sparsity,
    )
    full_solve_time = time.perf_counter() - start
    print(f"Full solve time: {full_solve_time:.3f} s")

    rhs_snapshots = np.empty_like(full_solution)
    for column, t in enumerate(times):
        rhs_snapshots[:, column] = full_rhs(t, full_solution[:, column])

    print("Computing solution and right-hand-side SVDs...")
    state_modes, state_values = svd_basis(full_solution)
    rhs_modes, rhs_values = svd_basis(rhs_snapshots)

    results = {
        "POD": [],
        "DEIM": [],
        "QDEIM": [],
        "SNS-DEIM": [],
        "SNS-QDEIM": [],
    }
    solve_times = {name: [] for name in results}
    condition_numbers = {name: [] for name in results if name != "POD"}
    display_trajectories = {}
    display_selections = {}
    local_validation_errors = {}

    for dimension in dimensions:
        print(f"Running reduced dimension {dimension}...")
        state_basis = state_modes[:, :dimension]
        rhs_basis = rhs_modes[:, :dimension]

        start = time.perf_counter()
        pod_coordinates = solve_pod(
            state_basis,
            args.final_time,
            times,
        )
        solve_times["POD"].append(time.perf_counter() - start)
        pod_solution = state_basis @ pod_coordinates
        results["POD"].append(relative_error(full_solution, pod_solution))

        variants = {
            "DEIM": (rhs_basis, deim_indices),
            "QDEIM": (rhs_basis, qdeim_indices),
            "SNS-DEIM": (state_basis, deim_indices),
            "SNS-QDEIM": (state_basis, qdeim_indices),
        }

        current_trajectories = {"POD": pod_solution}
        current_selections = {}

        for name, (basis, selector) in variants.items():
            start = time.perf_counter()
            coordinates, indices, edges, condition_number = solve_hyper_reduced(
                state_basis,
                basis,
                selector,
                args.final_time,
                times,
            )
            solve_times[name].append(time.perf_counter() - start)

            reduced_solution = state_basis @ coordinates
            results[name].append(relative_error(full_solution, reduced_solution))
            condition_numbers[name].append(condition_number)

            validation_error, validation_edges = validate_local_evaluator(
                state_basis,
                indices,
            )
            if not np.array_equal(edges, validation_edges):
                raise RuntimeError("Inconsistent edge closure.")
            if validation_error > 1.0e-12:
                raise RuntimeError(
                    f"Local evaluator failed for {name}: {validation_error:.3e}"
                )

            if dimension == args.display_dimension:
                current_trajectories[name] = reduced_solution
                current_selections[name] = {
                    "indices": indices,
                    "edges": edges,
                }
                local_validation_errors[name] = validation_error

        if dimension == args.display_dimension:
            display_trajectories = current_trajectories
            display_selections = current_selections

        print(
            "  "
            + ", ".join(
                f"{name} {results[name][-1]:.3e}"
                for name in results
            )
        )

    plot_singular_values(output_directory, state_values, rhs_values)
    plot_errors(output_directory, dimensions, results)
    plot_node_voltage(
        output_directory,
        times,
        full_solution,
        display_trajectories,
        args.display_dimension,
    )
    plot_sampling_stencils(
        output_directory,
        display_selections,
        args.n,
    )

    display_metrics = {}
    for name, trajectory in display_trajectories.items():
        key = name.lower().replace("-", "_")
        display_metrics[f"{key}_node1_error"] = (
            np.linalg.norm(full_solution[0, :] - trajectory[0, :])
            / np.linalg.norm(full_solution[0, :])
        )

    for name, data in display_selections.items():
        key = name.lower().replace("-", "_")
        required_nodes = set()
        for edge in data["edges"]:
            if edge == 0:
                required_nodes.add(0)
            else:
                required_nodes.add(int(edge - 1))
                required_nodes.add(int(edge))
        display_metrics[f"{key}_indices"] = data["indices"]
        display_metrics[f"{key}_required_edges"] = data["edges"]
        display_metrics[f"{key}_required_node_count"] = len(required_nodes)

    np.savez(
        output_directory / "rc_ladder_results.npz",
        n=args.n,
        final_time=args.final_time,
        snapshots=args.snapshots,
        dimensions=np.array(dimensions),
        times=times,
        full_solution=full_solution,
        state_singular_values=state_values,
        rhs_singular_values=rhs_values,
        full_solve_time=full_solve_time,
        **{
            f"{name.lower().replace('-', '_')}_errors": np.array(values)
            for name, values in results.items()
        },
        **{
            f"{name.lower().replace('-', '_')}_times": np.array(values)
            for name, values in solve_times.items()
        },
        **{
            f"{name.lower().replace('-', '_')}_condition_numbers": np.array(values)
            for name, values in condition_numbers.items()
        },
        **display_metrics,
    )

    print("Local evaluator validation at the displayed dimension:")
    for name, error in local_validation_errors.items():
        print(f"  {name}: {error:.3e}")
    print("Displayed-dimension node-1 errors and stencil sizes:")
    for name in display_trajectories:
        key = name.lower().replace("-", "_")
        message = (
            f"  {name}: node-1 error "
            f"{display_metrics[f'{key}_node1_error']:.3e}"
        )
        if name in display_selections:
            message += (
                f", {display_metrics[f'{key}_required_node_count']} required nodes"
            )
        print(message)
    print(f"Results written to {output_directory}")


if __name__ == "__main__":
    main()
