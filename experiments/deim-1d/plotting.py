from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_snapshots(
    x,
    function,
    parameters,
    *,
    deim_indices=None,
    title="Snapshots of the nonlinear function",
    save_name=None,
    show=True,
):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for parameter in parameters:
        values = function(x, parameter)
        ax.plot(x, values, linewidth=1.8, label=rf"$p={parameter:.2f}$")

    if deim_indices is not None:
        deim_indices = np.asarray(deim_indices, dtype=int)
        ax.scatter(
            x[deim_indices],
            function(x[deim_indices], parameters[-1]),
            color="black",
            s=32,
            zorder=5,
            label="DEIM points",
        )

    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x;p)$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    if save_name is not None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / save_name, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_deim_approximation(
    x,
    exact_values_list,
    approximation_values_list,
    deim_indices,
    *,
    parameters,
    save_name=None,
    show=True,
):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    axes = axes.ravel()

    for ax, parameter, exact_values, approximation_values in zip(
        axes,
        parameters,
        exact_values_list,
        approximation_values_list,
    ):
        ax.plot(x, exact_values, linewidth=1.8, label="Exact")
        ax.plot(x, approximation_values, "--", linewidth=1.8, label="DEIM")
        ax.scatter(
            x[deim_indices],
            exact_values[deim_indices],
            color="black",
            s=24,
            zorder=5,
            label="DEIM points",
        )

        ax.set_title(rf"$p={parameter:.2f}$")
        ax.set_ylabel(r"$f(x;p)$")
        ax.grid(alpha=0.25)

    for ax in axes[-2:]:
        ax.set_xlabel(r"$x$")

    axes[0].legend(fontsize=9)
    m = len(deim_indices)
    fig.suptitle(f"DEIM approximation for unseen parameters, m={m}")
    fig.tight_layout()

    if save_name is not None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / save_name, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
