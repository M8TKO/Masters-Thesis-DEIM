import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from common import (
    EXTERNAL_U,
    HOLDOUT_U,
    MODES,
    SEED,
    align_validation_to_reference,
    basis_orthonormality,
    environment_summary,
    error_metrics,
    greedy_deim_indices,
    interpolation_residual,
    load_dataset,
    nearest_run,
    pod_project,
    qdeim_indices,
    randomized_pod_basis,
    roundtrip_interpolation_control,
    selection_matrix_diagnostics,
    sparse_reconstruct,
    stack_runs,
    timed_call,
    write_csv,
    write_json,
    write_matrix_series,
)


def save_sensor_coordinates(path, coordinates, indices):
    rows = ["index,x,y"]
    for index in indices:
        x, y = coordinates[index, :2]
        rows.append(f"{int(index)},{x:.16e},{y:.16e}")
    Path(path).write_text("\n".join(rows) + "\n")


def run_mode_sweep(dataset_root, results_dir):
    metadata, runs = load_dataset(dataset_root, mmap=True)
    holdout = nearest_run(runs, HOLDOUT_U)
    training_runs = [run for run in runs if run is not holdout]

    if abs(holdout["u_max"] - HOLDOUT_U) > 1.0e-12:
        raise ValueError(f"Could not find holdout u={HOLDOUT_U}.")

    training_matrix = stack_runs(training_runs)
    truth = np.asarray(holdout["matrix"])
    max_modes = max(MODES)

    (mean, basis, singular_values), pod_time = timed_call(
        randomized_pod_basis,
        training_matrix,
        max_modes,
        10,
        1,
        SEED,
    )

    np.save(results_dir / "pod_mean.npy", mean)
    np.save(results_dir / "pod_basis_m200.npy", basis)
    np.save(results_dir / "singular_values.npy", singular_values)

    rows = []
    by_mode = {}
    for m in MODES:
        basis_m = basis[:, :m]
        mode_report = {
            "m": m,
            "pod_orthonormality_2_norm": basis_orthonormality(basis_m),
            "methods": {},
        }

        projection, reconstruction_time = timed_call(pod_project, mean, basis_m, truth)
        metrics, time_errors = error_metrics(truth, projection)
        np.save(results_dir / f"pod_projection_time_errors_m{m}.npy", time_errors)
        row = {
            "experiment": "leave_one_out",
            "method": "POD_projection",
            "m": m,
            "num_sensors": "",
            **metrics,
            "condition_2": "",
            "inverse_norm_2": "",
            "sensor_selection_time_seconds": 0.0,
            "reconstruction_time_seconds": reconstruction_time,
            "pod_orthonormality_2_norm": mode_report["pod_orthonormality_2_norm"],
            "distinct_indices": "",
            "nonsingular": "",
            "max_abs_interpolation_residual": "",
            "relative_interpolation_residual": "",
            "mass_l2_relative_error": "",
        }
        rows.append(row)
        mode_report["methods"]["POD_projection"] = row

        for method_name, selector, file_prefix in [
            ("DEIM", greedy_deim_indices, "deim"),
            ("QDEIM", qdeim_indices, "qdeim"),
        ]:
            indices, selection_time = timed_call(selector, basis_m)
            np.save(results_dir / f"{file_prefix}_indices_m{m}.npy", indices)
            reconstruction, reconstruction_time = timed_call(
                sparse_reconstruct, mean, basis_m, indices, truth
            )
            metrics, time_errors = error_metrics(truth, reconstruction)
            diagnostics = selection_matrix_diagnostics(basis_m, indices)
            residuals = interpolation_residual(mean, basis_m, indices, truth, reconstruction)
            np.save(results_dir / f"{file_prefix}_time_errors_m{m}.npy", time_errors)
            row = {
                "experiment": "leave_one_out",
                "method": method_name,
                "m": m,
                "num_sensors": int(indices.size),
                **metrics,
                "condition_2": diagnostics["condition_2"],
                "inverse_norm_2": diagnostics["inverse_norm_2"],
                "sensor_selection_time_seconds": selection_time,
                "reconstruction_time_seconds": reconstruction_time,
                "pod_orthonormality_2_norm": mode_report["pod_orthonormality_2_norm"],
                "distinct_indices": diagnostics["distinct_indices"],
                "nonsingular": diagnostics["nonsingular"],
                **residuals,
                "mass_l2_relative_error": "",
            }
            rows.append(row)
            mode_report["methods"][method_name] = row

            if m == 150:
                save_sensor_coordinates(
                    results_dir / f"{file_prefix}_sensor_coordinates_m150.csv",
                    holdout["coordinates"],
                    indices,
                )

        by_mode[str(m)] = mode_report

    training_u_values = [run["u_max"] for run in training_runs]
    mode_report = {
        "experiment": "leave_one_parameter_out",
        "dataset_root": str(dataset_root),
        "holdout_u": holdout["u_max"],
        "training_u_values": training_u_values,
        "training_matrix_shape": list(training_matrix.shape),
        "holdout_matrix_shape": list(truth.shape),
        "modes": MODES,
        "random_seed": SEED,
        "randomized_pod_seconds": pod_time,
        "no_holdout_snapshots_used": all(abs(u - HOLDOUT_U) > 1.0e-12 for u in training_u_values),
        "same_nested_basis_for_all_m": True,
        "mass_l2_error": {
            "computed": False,
            "reason": "The stored NumPy vorticity arrays are used as the primary DEIM state space. Reconstructing a matching finite-element mass matrix from these saved arrays was not used for the authoritative metrics.",
        },
        "singular_values_file": "singular_values.npy",
        "modes_detail": by_mode,
        "dataset_metadata": metadata,
    }

    csv_fields = [
        "experiment",
        "method",
        "m",
        "num_sensors",
        "relative_frobenius_error",
        "mean_snapshot_relative_error",
        "median_snapshot_relative_error",
        "p95_snapshot_relative_error",
        "max_snapshot_relative_error",
        "min_snapshot_relative_error",
        "condition_2",
        "inverse_norm_2",
        "sensor_selection_time_seconds",
        "reconstruction_time_seconds",
        "pod_orthonormality_2_norm",
        "distinct_indices",
        "nonsingular",
        "max_abs_interpolation_residual",
        "relative_interpolation_residual",
        "mass_l2_relative_error",
    ]
    write_csv(results_dir / "mode_sweep.csv", rows, csv_fields)
    write_json(results_dir / "mode_sweep.json", mode_report)
    return mode_report


def run_external_validation(dataset_root, validation_root, results_dir, external_fields_dir, paraview_dir, selected_m):
    metadata, training_runs = load_dataset(dataset_root, mmap=True)
    validation_metadata, validation_runs = load_dataset(validation_root, mmap=False)
    if len(validation_runs) != 1:
        raise ValueError("External validation root must contain exactly one run.")

    validation = validation_runs[0]
    if abs(validation["u_max"] - EXTERNAL_U) > 1.0e-12:
        raise ValueError(f"External validation parameter is {validation['u_max']}, expected {EXTERNAL_U}.")
    if any(abs(run["u_max"] - EXTERNAL_U) < 1.0e-12 for run in training_runs):
        raise ValueError("External validation parameter is part of the training grid.")

    training_matrix = stack_runs(training_runs)
    reference_run = training_runs[0]
    reference_coordinates = reference_run["coordinates"]
    aligned_truth, alignment_info = align_validation_to_reference(validation, reference_coordinates)
    roundtrip = roundtrip_interpolation_control(reference_run, validation)

    (mean, basis, singular_values), pod_time = timed_call(
        randomized_pod_basis,
        training_matrix,
        selected_m,
        10,
        1,
        SEED,
    )

    np.save(results_dir / "external_pod_mean.npy", mean)
    np.save(results_dir / f"external_pod_basis_m{selected_m}.npy", basis)
    np.save(results_dir / "external_singular_values.npy", singular_values)

    external_fields_dir.mkdir(parents=True, exist_ok=True)
    np.save(external_fields_dir / "truth.npy", aligned_truth)
    np.save(external_fields_dir / "times.npy", validation["times"])

    methods = {}
    for method_name, selector, file_prefix in [
        ("DEIM", greedy_deim_indices, "deim"),
        ("QDEIM", qdeim_indices, "qdeim"),
    ]:
        indices, selection_time = timed_call(selector, basis)
        np.save(results_dir / f"external_{file_prefix}_indices_m{selected_m}.npy", indices)
        reconstruction, reconstruction_time = timed_call(
            sparse_reconstruct, mean, basis, indices, aligned_truth
        )
        error = np.abs(reconstruction - aligned_truth)
        metrics, time_errors = error_metrics(aligned_truth, reconstruction)
        diagnostics = selection_matrix_diagnostics(basis, indices)
        residuals = interpolation_residual(mean, basis, indices, aligned_truth, reconstruction)

        np.save(external_fields_dir / f"{file_prefix}_reconstruction.npy", reconstruction)
        np.save(external_fields_dir / f"{file_prefix}_error.npy", error)
        np.save(results_dir / f"external_{file_prefix}_time_errors_m{selected_m}.npy", time_errors)

        methods[method_name] = {
            "m": selected_m,
            "num_sensors": int(indices.size),
            **metrics,
            **diagnostics,
            **residuals,
            "sensor_selection_time_seconds": selection_time,
            "reconstruction_time_seconds": reconstruction_time,
        }

    reference_h5 = reference_run["h5"]
    paraview_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_series(reference_h5, paraview_dir / "truth_vorticity.xdmf", aligned_truth, validation["times"], "truth_vorticity")
    write_matrix_series(reference_h5, paraview_dir / "deim_vorticity.xdmf", np.load(external_fields_dir / "deim_reconstruction.npy"), validation["times"], "deim_vorticity")
    write_matrix_series(reference_h5, paraview_dir / "qdeim_vorticity.xdmf", np.load(external_fields_dir / "qdeim_reconstruction.npy"), validation["times"], "qdeim_vorticity")
    write_matrix_series(reference_h5, paraview_dir / "deim_error.xdmf", np.load(external_fields_dir / "deim_error.npy"), validation["times"], "deim_error")
    write_matrix_series(reference_h5, paraview_dir / "qdeim_error.xdmf", np.load(external_fields_dir / "qdeim_error.npy"), validation["times"], "qdeim_error")

    report = {
        "experiment": "external_unseen_parameter_validation",
        "dataset_root": str(dataset_root),
        "validation_root": str(validation_root),
        "training_u_values": [run["u_max"] for run in training_runs],
        "external_u": validation["u_max"],
        "external_parameter_not_in_training_grid": True,
        "training_matrix_shape": list(training_matrix.shape),
        "original_validation_matrix_shape": list(validation["matrix"].shape),
        "aligned_validation_matrix_shape": list(aligned_truth.shape),
        "selected_m": selected_m,
        "random_seed": SEED,
        "randomized_pod_seconds": pod_time,
        "pod_orthonormality_2_norm": basis_orthonormality(basis),
        "same_basis_and_mean_for_deim_and_qdeim": True,
        "mesh_alignment": alignment_info,
        "roundtrip_interpolation_control": roundtrip,
        "methods": methods,
        "training_metadata": metadata,
        "validation_metadata": validation_metadata,
        "paraview_reference_h5": str(reference_h5),
    }
    write_json(results_dir / "external_validation.json", report)
    return report


def write_environment(output_root):
    env = environment_summary()
    conda = shutil.which("conda")
    if conda:
        try:
            env_prefix = str(Path(env["executable"]).parents[1])
            env["conda_list"] = subprocess.check_output(
                [conda, "list", "-p", env_prefix], text=True, timeout=30
            )
            env["conda_list_prefix"] = env_prefix
        except Exception as exc:
            env["conda_list_error"] = str(exc)
    lines = [
        f"Python executable: {env['executable']}",
        f"Python version: {env['python']}",
        f"Platform: {env['platform']}",
        "",
        "Packages:",
    ]
    for name, version in env["packages"].items():
        lines.append(f"- {name}: {version}")
    if "conda_list" in env:
        lines.extend(["", f"conda list -p {env['conda_list_prefix']}:", "", env["conda_list"]])
    elif "conda_list_error" in env:
        lines.extend(["", f"conda list unavailable: {env['conda_list_error']}"])
    (output_root / "environment.txt").write_text("\n".join(lines))
    return env


def main():
    parser = argparse.ArgumentParser(description="Run thesis POD/DEIM/QDEIM experiments.")
    parser.add_argument("--dataset-root", default="dataset_u_max_sweep_dense")
    parser.add_argument("--validation-root", default="validation_u_max_0_855")
    parser.add_argument("--output-root", default="thesis_navier_stokes_deim_qdeim")
    parser.add_argument("--selected-m", type=int, default=150)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results_dir = output_root / "results"
    external_fields_dir = output_root / "external_fields"
    paraview_dir = output_root / "paraview"
    for path in [results_dir, external_fields_dir, paraview_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    env = write_environment(output_root)
    configuration = {
        "dataset_root": args.dataset_root,
        "validation_root": args.validation_root,
        "output_root": args.output_root,
        "modes": MODES,
        "holdout_u": HOLDOUT_U,
        "external_u": EXTERNAL_U,
        "selected_external_m": args.selected_m,
        "random_seed": SEED,
        "environment": env,
    }
    write_json(results_dir / "configuration.json", configuration)

    mode_report = run_mode_sweep(Path(args.dataset_root), results_dir)
    external_report = run_external_validation(
        Path(args.dataset_root),
        Path(args.validation_root),
        results_dir,
        external_fields_dir,
        paraview_dir,
        args.selected_m,
    )

    elapsed = time.perf_counter() - start
    summary = {
        "total_runtime_seconds": elapsed,
        "mode_sweep": {
            "holdout_u": mode_report["holdout_u"],
            "training_matrix_shape": mode_report["training_matrix_shape"],
        },
        "external_validation": {
            "external_u": external_report["external_u"],
            "selected_m": external_report["selected_m"],
            "methods": external_report["methods"],
        },
    }
    write_json(results_dir / "run_summary.json", summary)
    print(f"Experiment completed in {elapsed:.2f} seconds.")
    print(f"Results written to {output_root}")


if __name__ == "__main__":
    main()
