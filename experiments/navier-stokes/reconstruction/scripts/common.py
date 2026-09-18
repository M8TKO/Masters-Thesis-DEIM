import csv
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.linalg import qr
from scipy.spatial import cKDTree


SEED = 1234
MODES = [20, 30, 40, 60, 80, 100, 120, 150, 200]
HOLDOUT_U = 0.85
EXTERNAL_U = 0.855


def parameter_label(value):
    return f"{value:.3f}".replace(".", "_")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(dataset_root, mmap=True):
    dataset_root = Path(dataset_root)
    metadata = read_json(dataset_root / "metadata.json")
    runs = []
    for run in metadata["runs"]:
        run_dir = dataset_root / run["directory"]
        matrix_kwargs = {"mmap_mode": "r"} if mmap else {}
        runs.append(
            {
                "u_max": float(run["u_max"]),
                "directory": run["directory"],
                "run_dir": run_dir,
                "matrix": np.load(run_dir / "vorticity_matrix.npy", **matrix_kwargs),
                "times": np.load(run_dir / "times.npy"),
                "coordinates": np.load(run_dir / "coordinates.npy"),
                "h5": run_dir / f"wake_u_max_{parameter_label(float(run['u_max']))}.h5",
            }
        )
    return metadata, runs


def stack_runs(runs):
    return np.column_stack([np.asarray(run["matrix"]) for run in runs])


def nearest_run(runs, target_u):
    return min(runs, key=lambda run: abs(run["u_max"] - target_u))


def randomized_pod_basis(snapshot_matrix, num_modes, oversampling=10, power_iterations=1, seed=SEED):
    mean = snapshot_matrix.mean(axis=1, keepdims=True)
    centered = snapshot_matrix - mean

    target_rank = min(num_modes + oversampling, centered.shape[0], centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], target_rank))

    y = centered @ omega
    for _ in range(power_iterations):
        y = centered @ (centered.T @ y)

    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ centered
    u_small, singular_values, _ = np.linalg.svd(b, full_matrices=False)
    basis = q @ u_small[:, :num_modes]
    return mean[:, 0], basis, singular_values


def greedy_deim_indices(basis):
    indices = [int(np.argmax(np.abs(basis[:, 0])))]
    selected_basis = basis[:, [0]]

    for mode in range(1, basis.shape[1]):
        interpolation_matrix = selected_basis[indices, :]
        coefficients = np.linalg.solve(interpolation_matrix, basis[indices, mode])
        residual = basis[:, mode] - selected_basis @ coefficients
        indices.append(int(np.argmax(np.abs(residual))))
        selected_basis = basis[:, : mode + 1]

    return np.array(indices, dtype=np.int64)


def qdeim_indices(basis):
    _, _, pivots = qr(basis.T, pivoting=True, mode="economic")
    return np.asarray(pivots[: basis.shape[1]], dtype=np.int64)


def pod_project(mean, basis, truth):
    centered_truth = truth - mean[:, None]
    return mean[:, None] + basis @ (basis.T @ centered_truth)


def sparse_reconstruct(mean, basis, indices, truth):
    sensor_basis = basis[indices, :]
    sensor_truth = truth[indices, :] - mean[indices, None]
    coefficients = np.linalg.solve(sensor_basis, sensor_truth)
    return mean[:, None] + basis @ coefficients


def error_metrics(truth, approximation):
    diff = truth - approximation
    truth_norm = np.linalg.norm(truth, "fro")
    column_norms = np.linalg.norm(truth, axis=0)
    column_errors = np.linalg.norm(diff, axis=0) / column_norms
    return {
        "relative_frobenius_error": float(np.linalg.norm(diff, "fro") / truth_norm),
        "mean_snapshot_relative_error": float(column_errors.mean()),
        "median_snapshot_relative_error": float(np.median(column_errors)),
        "p95_snapshot_relative_error": float(np.percentile(column_errors, 95)),
        "max_snapshot_relative_error": float(column_errors.max()),
        "min_snapshot_relative_error": float(column_errors.min()),
    }, column_errors


def interpolation_residual(mean, basis, indices, truth, reconstruction):
    residual = reconstruction[indices, :] - truth[indices, :]
    reference = truth[indices, :]
    return {
        "max_abs_interpolation_residual": float(np.max(np.abs(residual))),
        "relative_interpolation_residual": float(
            np.linalg.norm(residual, "fro") / np.linalg.norm(reference, "fro")
        ),
    }


def selection_matrix_diagnostics(basis, indices):
    interpolation_matrix = basis[indices, :]
    return {
        "distinct_indices": int(np.unique(indices).size),
        "condition_2": float(np.linalg.cond(interpolation_matrix)),
        "inverse_norm_2": float(np.linalg.norm(np.linalg.solve(interpolation_matrix, np.eye(interpolation_matrix.shape[0])), 2)),
        "nonsingular": bool(np.linalg.matrix_rank(interpolation_matrix) == interpolation_matrix.shape[0]),
    }


def basis_orthonormality(basis):
    m = basis.shape[1]
    return float(np.linalg.norm(basis.T @ basis - np.eye(m), 2))


def align_validation_to_reference(validation_run, reference_coordinates):
    source_coordinates = validation_run["coordinates"]
    source_matrix = np.asarray(validation_run["matrix"])
    source_xy = source_coordinates[:, :2]
    target_xy = reference_coordinates[:, :2]

    linear = LinearNDInterpolator(source_xy, source_matrix)
    mapped = linear(target_xy)
    fallback_mask = np.isnan(mapped).any(axis=1)
    fallback_count = int(fallback_mask.sum())
    max_fallback_distance = 0.0

    if fallback_count:
        nearest = NearestNDInterpolator(source_xy, source_matrix)
        nearest_values = nearest(target_xy[fallback_mask])
        mapped[fallback_mask, :] = nearest_values
        tree = cKDTree(source_xy)
        distances, _ = tree.query(target_xy[fallback_mask])
        max_fallback_distance = float(distances.max())

    return np.asarray(mapped), {
        "method": "linear_interpolation_with_nearest_neighbour_fallback",
        "source_dofs": int(source_coordinates.shape[0]),
        "target_dofs": int(reference_coordinates.shape[0]),
        "nearest_fallback_target_points": fallback_count,
        "max_nearest_fallback_distance": max_fallback_distance,
    }


def roundtrip_interpolation_control(reference_run, validation_run):
    reference_coordinates = reference_run["coordinates"]
    validation_coordinates = validation_run["coordinates"]
    reference_matrix = np.asarray(reference_run["matrix"])

    to_validation, first = align_validation_to_reference(
        {
            "coordinates": reference_coordinates,
            "matrix": reference_matrix,
        },
        validation_coordinates,
    )
    back_to_reference, second = align_validation_to_reference(
        {
            "coordinates": validation_coordinates,
            "matrix": to_validation,
        },
        reference_coordinates,
    )
    metrics, _ = error_metrics(reference_matrix, back_to_reference)
    return {
        "reference_u": float(reference_run["u_max"]),
        "reference_shape": list(reference_matrix.shape),
        "validation_grid_shape": list(to_validation.shape),
        "reference_to_validation": first,
        "validation_to_reference": second,
        "relative_frobenius_error": metrics["relative_frobenius_error"],
        "mean_snapshot_relative_error": metrics["mean_snapshot_relative_error"],
        "p95_snapshot_relative_error": metrics["p95_snapshot_relative_error"],
        "max_snapshot_relative_error": metrics["max_snapshot_relative_error"],
    }


def hdf5_time_name(time_value):
    return f"{time_value:.17g}".replace(".", "_")


def copy_mesh_from_original(original_h5, output_h5):
    with h5py.File(original_h5, "r") as source, h5py.File(output_h5, "w") as target:
        source.copy("/Mesh", target)
        topology = target["/Mesh/mesh/topology"][:]
        geometry = target["/Mesh/mesh/geometry"][:]
    return topology.shape[0], geometry.shape[0]


def write_field_h5(output_h5, field_name, matrix, times):
    with h5py.File(output_h5, "a") as h5:
        group = h5.require_group(f"/Function/{field_name}")
        for column, time_value in enumerate(times):
            group.create_dataset(
                hdf5_time_name(time_value),
                data=matrix[:, column].reshape((-1, 1)),
            )


def write_xdmf(output_xdmf, output_h5_name, field_name, times, num_cells, num_nodes):
    lines = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">',
        "  <Domain>",
        '    <Grid Name="mesh" GridType="Uniform">',
        f'      <Topology TopologyType="Triangle" NumberOfElements="{num_cells}" NodesPerElement="3">',
        f'        <DataItem Dimensions="{num_cells} 3" NumberType="Int" Format="HDF">{output_h5_name}:/Mesh/mesh/topology</DataItem>',
        "      </Topology>",
        '      <Geometry GeometryType="XY">',
        f'        <DataItem Dimensions="{num_nodes} 2" Format="HDF">{output_h5_name}:/Mesh/mesh/geometry</DataItem>',
        "      </Geometry>",
        "    </Grid>",
        f'    <Grid Name="{field_name}" GridType="Collection" CollectionType="Temporal">',
    ]
    for time_value in times:
        time_name = hdf5_time_name(time_value)
        lines.extend(
            [
                f'      <Grid Name="{field_name}" GridType="Uniform">',
                '        <xi:include xpointer="xpointer(/Xdmf/Domain/Grid[@GridType=\'Uniform\'][1]/*[self::Topology or self::Geometry])" />',
                f'        <Time Value="{time_value:.17g}" />',
                f'        <Attribute Name="{field_name}" AttributeType="Scalar" Center="Node">',
                f'          <DataItem Dimensions="{num_nodes} 1" Format="HDF">{output_h5_name}:/Function/{field_name}/{time_name}</DataItem>',
                "        </Attribute>",
                "      </Grid>",
            ]
        )
    lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>", ""])
    Path(output_xdmf).write_text("\n".join(lines))


def write_matrix_series(original_h5, output_xdmf, matrix, times, field_name):
    output_xdmf = Path(output_xdmf)
    output_h5 = output_xdmf.with_suffix(".h5")
    num_cells, num_nodes = copy_mesh_from_original(original_h5, output_h5)
    if matrix.shape != (num_nodes, len(times)):
        raise ValueError(f"{field_name}: expected {(num_nodes, len(times))}, got {matrix.shape}.")
    write_field_h5(output_h5, field_name, matrix, times)
    write_xdmf(output_xdmf, output_h5.name, field_name, times, num_cells, num_nodes)


def load_mesh_arrays(original_h5):
    with h5py.File(original_h5, "r") as h5:
        topology = h5["/Mesh/mesh/topology"][:]
        geometry = h5["/Mesh/mesh/geometry"][:]
    return topology, geometry


def environment_summary():
    packages = {}
    for name in ["numpy", "scipy", "matplotlib", "h5py"]:
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "unknown")
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    value = func(*args, **kwargs)
    return value, time.perf_counter() - start
