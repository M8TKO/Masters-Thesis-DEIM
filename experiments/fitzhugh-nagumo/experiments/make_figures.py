from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS = PROJECT_ROOT / "results" / "results.npz"
OUTPUT = PROJECT_ROOT / "figures"


def main():
    data = np.load(RESULTS)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    x = data["x"]
    t = data["t"]
    y_full = data["Y_full"]
    y_rom = data["Y_rom"]
    state_svals = data["state_svals"]
    nonlinear_svals = data["nonlin_svals"]
    timewise_error = data["timewise_error"]
    n = len(x)
    node = n // 2

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    modes = np.arange(1, len(state_svals) + 1)
    ax.semilogy(modes, state_svals / state_svals[0], label="State snapshots")
    ax.semilogy(
        modes,
        nonlinear_svals / nonlinear_svals[0],
        label="Nonlinear snapshots",
    )
    ax.axvline(10, color="black", linestyle="--", linewidth=1.2, label="$k=m=10$")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Normalized singular value")
    ax.set_xlim(1, 60)
    ax.set_ylim(1e-13, 2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "fhn_snapshot_singular_values.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True)
    axes[0].plot(t, y_full[node, :], label="Full model")
    axes[0].plot(t, y_rom[node, :], "--", label="POD--DEIM")
    axes[0].set_ylabel(f"$v(x={x[node]:.3f},t)$")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    axes[1].plot(t, y_full[n + node, :], label="Full model")
    axes[1].plot(t, y_rom[n + node, :], "--", label="POD--DEIM")
    axes[1].set_ylabel(f"$w(x={x[node]:.3f},t)$")

    axes[2].semilogy(t, timewise_error)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Relative state error")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT / "fhn_rom_diagnostics.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
