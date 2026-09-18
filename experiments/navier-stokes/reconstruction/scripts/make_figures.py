import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

from common import (
    HOLDOUT_U,
    greedy_deim_indices,
    load_dataset,
    load_mesh_arrays,
    qdeim_indices,
    read_json,
    sparse_reconstruct,
    stack_runs,
    write_json,
)


matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def save_both(fig, output_stem):
    output_stem = Path(output_stem)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def numeric(value):
    if value == "":
        return np.nan
    return float(value)


def rows_for_method(rows, method):
    selected = [row for row in rows if row["method"] == method]
    selected.sort(key=lambda row: int(row["m"]))
    return selected


def exact_centered_energy(dataset_root):
    _, runs = load_dataset(dataset_root, mmap=True)
    training_runs = [run for run in runs if abs(run["u_max"] - HOLDOUT_U) > 1.0e-12]
    training_matrix = stack_runs(training_runs)
    mean = training_matrix.mean(axis=1, keepdims=True)
    centered = training_matrix - mean
    return float(np.linalg.norm(centered, "fro") ** 2), list(training_matrix.shape)


def plot_singular_values(results_dir, figures_dir, dataset_root):
    singular_values = np.load(results_dir / "singular_values.npy")
    total_centered_energy, training_shape = exact_centered_energy(dataset_root)
    normalized = singular_values / singular_values[0]
    energy = np.cumsum(singular_values**2) / total_centered_energy
    modes = np.arange(1, singular_values.size + 1)

    write_json(
        results_dir / "singular_value_decay.json",
        {
            "denominator": "exact_centered_training_matrix_frobenius_norm_squared",
            "centered_training_matrix_frobenius_norm_squared": total_centered_energy,
            "training_matrix_shape": training_shape,
            "computed_singular_value_energy_fraction": float(
                np.sum(singular_values**2) / total_centered_energy
            ),
            "last_cumulative_energy_value": float(energy[-1]),
        },
    )

    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.semilogy(modes, normalized, color="#1f77b4", linewidth=1.8, label="normalized singular value")
    ax1.set_xlabel("mode index")
    ax1.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax1.grid(True, alpha=0.28)

    ax2 = ax1.twinx()
    ax2.plot(modes, energy, color="#d62728", linewidth=1.8, label="cumulative energy")
    ax2.set_ylabel("cumulative captured energy")
    ax2.set_ylim(0.0, 1.01)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    save_both(fig, figures_dir / "singular_value_decay")


def plot_error_vs_m(rows, figures_dir):
    styles = {
        "POD_projection": ("#111827", "o", "POD projection"),
        "DEIM": ("#d7191c", "s", "DEIM"),
        "QDEIM": ("#2c7bb6", "^", "QDEIM"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, (color, marker, label) in styles.items():
        method_rows = rows_for_method(rows, method)
        m = [int(row["m"]) for row in method_rows]
        e = [numeric(row["relative_frobenius_error"]) for row in method_rows]
        ax.plot(m, e, marker=marker, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("number of modes / sensors")
    ax.set_ylabel("relative Frobenius error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend()
    save_both(fig, figures_dir / "mode_sweep_frobenius_error")


def plot_time_error_vs_m(rows, figures_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, color, marker, label in [
        ("POD_projection", "#111827", "o", "POD projection"),
        ("DEIM", "#d7191c", "s", "DEIM"),
        ("QDEIM", "#2c7bb6", "^", "QDEIM"),
    ]:
        method_rows = rows_for_method(rows, method)
        m = [int(row["m"]) for row in method_rows]
        e = [numeric(row["p95_snapshot_relative_error"]) for row in method_rows]
        ax.plot(m, e, marker=marker, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("number of modes / sensors")
    ax.set_ylabel("95th-percentile snapshot-relative error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend()
    save_both(fig, figures_dir / "mode_sweep_time_local_error")


def plot_condition_vs_m(rows, figures_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, color, marker in [("DEIM", "#d7191c", "s"), ("QDEIM", "#2c7bb6", "^")]:
        method_rows = rows_for_method(rows, method)
        m = [int(row["m"]) for row in method_rows]
        cond = [numeric(row["condition_2"]) for row in method_rows]
        ax.plot(m, cond, marker=marker, color=color, linewidth=1.8, label=method)
    ax.set_xlabel("number of modes / sensors")
    ax.set_ylabel(r"$\mathrm{cond}_2(P^T U_m)$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend()
    save_both(fig, figures_dir / "interpolation_matrix_condition")


def draw_domain(ax):
    ax.add_patch(plt.Rectangle((0, 0), 2.2, 0.41, facecolor="#eef1f4", edgecolor="#1f2933", linewidth=0.9, zorder=0))
    ax.add_patch(plt.Circle((0.2, 0.2), 0.05, facecolor="#6b7280", edgecolor="#111827", linewidth=0.8, zorder=4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="white", linewidth=0.7)


def plot_sensor_locations(results_dir, figures_dir, dataset_root, zoom=False):
    _, runs = load_dataset(dataset_root, mmap=True)
    coordinate_run = next(run for run in runs if abs(run["u_max"] - 0.85) < 1.0e-12)
    coordinates = coordinate_run["coordinates"][:, :2]
    deim_indices = np.load(results_dir / "deim_indices_m150.npy")
    qdeim_indices = np.load(results_dir / "qdeim_indices_m150.npy")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), sharex=True, sharey=True)
    for ax, indices, color, label in [
        (axes[0], deim_indices, "#d7191c", "DEIM"),
        (axes[1], qdeim_indices, "#2c7bb6", "QDEIM"),
    ]:
        draw_domain(ax)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=0.8, c="#c7cdd4", alpha=0.24, linewidths=0, zorder=1)
        ax.scatter(coordinates[indices, 0], coordinates[indices, 1], s=16 if not zoom else 22, c=color, edgecolors="white", linewidths=0.35, zorder=5)
        ax.set_title(label, loc="left", pad=8, fontsize=11, fontweight="bold")
        if zoom:
            ax.set_xlim(0.04, 1.20)
            ax.set_ylim(0.02, 0.39)
        else:
            ax.set_xlim(-0.02, 2.22)
            ax.set_ylim(-0.02, 0.43)
    name = "sensor_locations_m150_wake_zoom" if zoom else "sensor_locations_m150_domain"
    save_both(fig, figures_dir / name)


def plot_external_error_time(results_dir, figures_dir):
    report = read_json(results_dir / "external_validation.json")
    times = np.load(Path(results_dir).parent / "external_fields" / "times.npy")
    deim = np.load(results_dir / "external_deim_time_errors_m150.npy")
    qdeim = np.load(results_dir / "external_qdeim_time_errors_m150.npy")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(times, deim, color="#d7191c", linewidth=1.5, label="DEIM")
    ax.plot(times, qdeim, color="#2c7bb6", linewidth=1.5, label="QDEIM")
    ax.set_xlabel("time")
    ax.set_ylabel("snapshot-relative error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend()
    worse_method = max(report["methods"], key=lambda name: report["methods"][name]["max_snapshot_relative_error"])
    ax.axvline(times[np.argmax(deim if worse_method == "DEIM" else qdeim)], color="#6b7280", linestyle="--", linewidth=1.0)
    save_both(fig, figures_dir / "external_validation_error_time")


def select_developed_wake_indices(times):
    preferred_times = [6.0, 8.0]
    return [int(np.argmin(np.abs(times - value))) for value in preferred_times]


def decorate_field_axis(ax):
    ax.add_patch(
        plt.Circle(
            (0.2, 0.2),
            0.05,
            facecolor="#6b7280",
            edgecolor="#111827",
            linewidth=0.5,
            zorder=5,
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.45)
    ax.set_ylim(0.04, 0.36)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0.1, 0.2, 0.3])


def plot_vorticity_panels(results_dir, figures_dir):
    output_root = Path(results_dir).parent
    external_dir = output_root / "external_fields"
    report = read_json(results_dir / "external_validation.json")
    topology, geometry = load_mesh_arrays(report["paraview_reference_h5"])
    tri = mtri.Triangulation(geometry[:, 0], geometry[:, 1], topology)

    truth = np.load(external_dir / "truth.npy")
    deim = np.load(external_dir / "deim_reconstruction.npy")
    qdeim = np.load(external_dir / "qdeim_reconstruction.npy")
    deim_error = np.load(external_dir / "deim_error.npy")
    qdeim_error = np.load(external_dir / "qdeim_error.npy")
    times = np.load(external_dir / "times.npy")
    representative = select_developed_wake_indices(times)

    vlim = np.percentile(np.abs(np.column_stack([truth[:, representative], deim[:, representative], qdeim[:, representative]])), 99.5)
    elim = np.percentile(np.column_stack([deim_error[:, representative], qdeim_error[:, representative]]), 99.5)

    fig, axes = plt.subplots(len(representative), 3, figsize=(8.0, 2.55 * len(representative)), sharex=True, sharey=True)
    if len(representative) == 1:
        axes = axes[None, :]
    labels = [
        "Full field",
        "DEIM reconstruction",
        "QDEIM reconstruction",
    ]
    matrices = [truth, deim, qdeim]
    for row, index in enumerate(representative):
        for col, (label, matrix) in enumerate(zip(labels, matrices)):
            ax = axes[row, col]
            image = ax.tripcolor(tri, matrix[:, index], shading="gouraud", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            decorate_field_axis(ax)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"t = {times[index]:.2f}")
            if row == len(representative) - 1:
                ax.set_xlabel("x")
            if col == 2:
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    save_both(fig, figures_dir / "external_validation_vorticity_fields")

    fig, axes = plt.subplots(len(representative), 2, figsize=(6.0, 2.55 * len(representative)), sharex=True, sharey=True)
    if len(representative) == 1:
        axes = axes[None, :]
    labels = ["|DEIM error|", "|QDEIM error|"]
    matrices = [deim_error, qdeim_error]
    for row, index in enumerate(representative):
        for col, (label, matrix) in enumerate(zip(labels, matrices)):
            ax = axes[row, col]
            image = ax.tripcolor(tri, matrix[:, index], shading="gouraud", cmap="magma", vmin=0, vmax=elim)
            decorate_field_axis(ax)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"t = {times[index]:.2f}")
            if row == len(representative) - 1:
                ax.set_xlabel("x")
            if col == 1:
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    save_both(fig, figures_dir / "external_validation_error_fields")


def plot_vorticity_wake_contrast(results_dir, figures_dir):
    output_root = Path(results_dir).parent
    external_dir = output_root / "external_fields"
    report = read_json(results_dir / "external_validation.json")
    topology, geometry = load_mesh_arrays(report["paraview_reference_h5"])
    tri = mtri.Triangulation(geometry[:, 0], geometry[:, 1], topology)

    truth = np.load(external_dir / "truth.npy")
    times = np.load(external_dir / "times.npy")
    representative = select_developed_wake_indices(times)
    mean = np.load(results_dir / "external_pod_mean.npy")
    basis = np.load(results_dir / "external_pod_basis_m150.npy")
    modes = [80, 100, 120, 150]

    displayed = {}
    for m in modes:
        basis_m = basis[:, :m]
        for method, selector in [
            ("DEIM", greedy_deim_indices),
            ("QDEIM", qdeim_indices),
        ]:
            indices = selector(basis_m)
            reconstruction = sparse_reconstruct(mean, basis_m, indices, truth)
            displayed[(method, m)] = reconstruction[:, representative]

    x = geometry[:, 0]
    y = geometry[:, 1]
    radius_from_cylinder = np.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2)
    wake_mask = (
        (x >= 0.25)
        & (x <= 1.45)
        & (y >= 0.04)
        & (y <= 0.36)
        & (radius_from_cylinder > 0.075)
    )
    wake_values = [truth[:, representative]]
    wake_values.extend(displayed.values())
    wake_values = np.abs(np.column_stack(wake_values))
    vlim = float(np.percentile(wake_values[wake_mask, :], 99.5))

    rows = [
        ("DEIM", 0),
        ("DEIM", 1),
        ("QDEIM", 0),
        ("QDEIM", 1),
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(modes) + 1,
        figsize=(10.5, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    titles = ["Full field"] + [f"$m={m}$" for m in modes]
    image = None
    for row, (method, time_position) in enumerate(rows):
        index = representative[time_position]
        matrices = [truth[:, index]]
        matrices.extend(displayed[(method, m)][:, time_position] for m in modes)
        for col, (title, values) in enumerate(zip(titles, matrices)):
            ax = axes[row, col]
            image = ax.tripcolor(
                tri,
                values,
                shading="gouraud",
                cmap="seismic",
                vmin=-vlim,
                vmax=vlim,
            )
            decorate_field_axis(ax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(
                    f"{method}\n$t={times[index]:.0f}$",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                    fontsize=9,
                )
            if row == len(rows) - 1:
                ax.set_xlabel("x")
    fig.colorbar(image, ax=axes, fraction=0.016, pad=0.015)

    write_json(
        results_dir / "wake_contrast_figure_scale.json",
        {
            "figure": "external_validation_vorticity_wake_contrast",
            "scale": "symmetric_vorticity_scale_from_wake_region_99_5_percentile",
            "vorticity_limit": vlim,
            "wake_region": {
                "x_min": 0.25,
                "x_max": 1.45,
                "y_min": 0.04,
                "y_max": 0.36,
                "minimum_distance_from_cylinder_center": 0.075,
            },
            "modes": modes,
            "times": [float(times[index]) for index in representative],
        },
    )
    output_stem = figures_dir / "external_validation_vorticity_wake_contrast"
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make thesis figures.")
    parser.add_argument("--output-root", default="thesis_navier_stokes_deim_qdeim")
    parser.add_argument("--dataset-root", default="dataset_u_max_sweep_dense")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(results_dir / "mode_sweep.csv")
    plot_singular_values(results_dir, figures_dir, Path(args.dataset_root))
    plot_error_vs_m(rows, figures_dir)
    plot_time_error_vs_m(rows, figures_dir)
    plot_condition_vs_m(rows, figures_dir)
    plot_sensor_locations(results_dir, figures_dir, Path(args.dataset_root), zoom=False)
    plot_sensor_locations(results_dir, figures_dir, Path(args.dataset_root), zoom=True)
    plot_external_error_time(results_dir, figures_dir)
    plot_vorticity_panels(results_dir, figures_dir)
    plot_vorticity_wake_contrast(results_dir, figures_dir)

    figure_files = sorted(path.name for path in figures_dir.iterdir() if path.suffix in {".png", ".pdf"})
    (figures_dir / "figure_manifest.txt").write_text("\n".join(figure_files) + "\n")
    print(f"Saved {len(figure_files)} figure files to {figures_dir}")


if __name__ == "__main__":
    main()
