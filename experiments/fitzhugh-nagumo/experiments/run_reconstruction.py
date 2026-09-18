import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from src.pde_model import (
    EPSILON,
    B as FHN_B,
    GAMMA,
    C,
    build_neumann_laplacian_1d,
    boundary_input_vector,
    initial_condition,
    solve_full_model,
)
from src.reduction import compute_pod_basis, deim, solve_reduced_model


PLOT_DPI = 300


def choose_modes(U: np.ndarray, s: np.ndarray, min_modes: int, max_modes: int | None = None) -> tuple[np.ndarray, int]:
    k_auto = U.shape[1]
    k = max(k_auto, min_modes)
    if max_modes is not None:
        k = min(k, max_modes, len(s))
    else:
        k = min(k, len(s))
    return U[:, :k], k


def singular_values(Y: np.ndarray) -> np.ndarray:
    return np.linalg.svd(Y, full_matrices=False, compute_uv=False)


def build_reduced_operators(
    A: np.ndarray,
    Y: np.ndarray,
    G: np.ndarray,
    n: int,
    epsilon: float,
    b: float,
    gamma: float,
    c: float,
    state_tol: float,
    nonlin_tol: float,
    min_state_modes: int,
    min_nonlin_modes: int,
    max_state_modes: int | None = None,
    max_nonlin_modes: int | None = None,
):
    V_auto, sY = compute_pod_basis(Y, tol=state_tol)
    V_k, k = choose_modes(V_auto, sY, min_state_modes, max_state_modes)
    U_auto, sG = compute_pod_basis(G, tol=nonlin_tol)
    U_m, m = choose_modes(U_auto, sG, min_nonlin_modes, max_nonlin_modes)
    A_r = V_k.T @ A @ V_k
    indices, P = deim(U_m)
    PTU = U_m[indices, :]
    cond_ptu = np.linalg.cond(PTU)
    B_deim = np.linalg.solve(PTU.T, (V_k.T @ U_m).T).T
    r_full = np.zeros(2 * n)
    r_full[:n] = (c / epsilon) * np.ones(n)
    r_full[n:] = c * np.ones(n)
    r_const = V_k.T @ r_full
    return {
        "V_k": V_k,
        "U_m": U_m,
        "A_r": A_r,
        "B_deim": B_deim,
        "indices": indices,
        "P": P,
        "r_const": r_const,
        "state_svals": sY,
        "nonlin_svals": sG,
        "state_svals_v": singular_values(Y[:n, :]),
        "state_svals_w": singular_values(Y[n:, :]),
        "nonlin_svals_v": singular_values(G[:n, :]),
        "nonlin_svals_w": singular_values(G[n:, :]),
        "k": k,
        "m": m,
        "cond_ptu": cond_ptu,
    }


def relative_error(Y_ref: np.ndarray, Y_approx: np.ndarray) -> float:
    denom = np.linalg.norm(Y_ref)
    if denom == 0:
        return np.linalg.norm(Y_approx)
    return np.linalg.norm(Y_ref - Y_approx) / denom


def pointwise_relative_errors(Y_ref: np.ndarray, Y_approx: np.ndarray) -> np.ndarray:
    errs = np.zeros(Y_ref.shape[1])
    for j in range(Y_ref.shape[1]):
        denom = np.linalg.norm(Y_ref[:, j])
        errs[j] = np.linalg.norm(Y_ref[:, j] - Y_approx[:, j]) / max(denom, 1e-14)
    return errs


def configure_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.28, which="both")
    ax.tick_params(axis="both", labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)


def save_figure(fig, outdir: Path, filename: str):
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_singular_values(outdir: Path, svals: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    modes = np.arange(1, len(svals) + 1)
    ax.semilogy(modes, svals, "o-", linewidth=2.2, markersize=5.2)
    configure_axis(ax, "Mode index", "Singular value", title)
    save_figure(fig, outdir, filename)


def make_plots(outdir: Path, x, t, Y_full, Y_rom, n, svals_v, svals_w, nonlin_svals_v, nonlin_svals_w, node_index=None):
    outdir.mkdir(parents=True, exist_ok=True)
    v_full = Y_full[:n, :]
    w_full = Y_full[n:, :]
    v_rom = Y_rom[:n, :]
    w_rom = Y_rom[n:, :]
    if node_index is None:
        node_index = n // 2
    plot_nodes = np.linspace(0, n - 1, 12, dtype=int)
    plot_singular_values(outdir, svals_v, "Singular Value Decay for Voltage Snapshots", "singular_values_v.png")
    plot_singular_values(outdir, svals_w, "Singular Value Decay for Recovery Snapshots", "singular_values_w.png")
    plot_singular_values(outdir, nonlin_svals_v, "Singular Value Decay for Nonlinear Voltage Block", "singular_values_nonlinear_v.png")
    plot_singular_values(outdir, nonlin_svals_w, "Singular Value Decay for Nonlinear Recovery Block", "singular_values_nonlinear_w.png")
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(np.arange(1, len(svals_v) + 1), svals_v, "o-", linewidth=2.2, markersize=5.0, label="v snapshots")
    ax.semilogy(np.arange(1, len(svals_w) + 1), svals_w, "s--", linewidth=2.2, markersize=5.0, label="w snapshots")
    configure_axis(ax, "Mode index", "Singular value", "Separate Singular Value Decay of v and w")
    ax.legend(fontsize=11, frameon=True)
    save_figure(fig, outdir, "singular_values_v_w_comparison.png")
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    ax[0].plot(t, v_full[node_index, :], linewidth=2.2, label="Full")
    ax[0].plot(t, v_rom[node_index, :], "--", linewidth=2.2, label="Reduced")
    configure_axis(ax[0], "", f"v(x={x[node_index]:.3f})", "Time Traces at One Spatial Node")
    ax[0].legend(fontsize=11, frameon=True)
    ax[1].plot(t, w_full[node_index, :], linewidth=2.2, label="Full")
    ax[1].plot(t, w_rom[node_index, :], "--", linewidth=2.2, label="Reduced")
    configure_axis(ax[1], "t", f"w(x={x[node_index]:.3f})", "")
    save_figure(fig, outdir, "time_traces_one_node.png")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(v_full[node_index, :], w_full[node_index, :], label="Full", linewidth=2.4)
    ax.plot(v_rom[node_index, :], w_rom[node_index, :], "--", label="Reduced", linewidth=2.4)
    configure_axis(ax, f"v(x={x[node_index]:.3f}, t)", f"w(x={x[node_index]:.3f}, t)", "Phase Portrait at One Spatial Node")
    ax.legend(fontsize=11, frameon=True)
    save_figure(fig, outdir, "phase_portrait_one_node.png")
    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    for j, idx in enumerate(plot_nodes):
        if j == 0:
            ax.plot(v_full[idx, :], w_full[idx, :], linewidth=1.35, label="Full")
            ax.plot(v_rom[idx, :], w_rom[idx, :], "--", linewidth=1.75, label="Reduced")
        else:
            ax.plot(v_full[idx, :], w_full[idx, :], linewidth=1.05)
            ax.plot(v_rom[idx, :], w_rom[idx, :], "--", linewidth=1.25)
    configure_axis(ax, "v(x,t)", "w(x,t)", "Phase Portraits at Multiple Spatial Nodes")
    ax.legend(fontsize=11, frameon=True)
    save_figure(fig, outdir, "phase_portrait_multi_node.png")
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    for j, idx in enumerate(plot_nodes):
        x_curve = np.full_like(t, x[idx], dtype=float)
        if j == 0:
            ax.plot(x_curve, v_full[idx, :], w_full[idx, :], linewidth=1.15, label="Full")
            ax.plot(x_curve, v_rom[idx, :], w_rom[idx, :], "--", linewidth=1.55, label="Reduced")
        else:
            ax.plot(x_curve, v_full[idx, :], w_full[idx, :], linewidth=0.95)
            ax.plot(x_curve, v_rom[idx, :], w_rom[idx, :], "--", linewidth=1.1)
    ax.set_xlabel("x", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("v(x,t)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_zlabel("w(x,t)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("3D Phase-Space Diagram Across Spatial Nodes", fontsize=15, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, frameon=True)
    save_figure(fig, outdir, "phase_space_3d_multi_node.png")


def main():
    parser = argparse.ArgumentParser(description="POD-DEIM reconstruction diagnostics for the 1D FitzHugh-Nagumo system.")
    parser.add_argument("--n", type=int, default=512, help="Spatial grid points per field")
    parser.add_argument("--tf", type=float, default=8.0, help="Final time")
    parser.add_argument("--num-snapshots", type=int, default=100, help="Number of saved snapshots")
    parser.add_argument("--state-tol", type=float, default=1e-2, help="Relative-difference tolerance for state POD")
    parser.add_argument("--nonlin-tol", type=float, default=1e-2, help="Relative-difference tolerance for nonlinear POD")
    parser.add_argument("--min-k", type=int, default=8, help="Minimum number of state POD modes")
    parser.add_argument("--min-m", type=int, default=8, help="Minimum number of nonlinear POD/DEIM modes")
    parser.add_argument("--max-k", type=int, default=1000, help="Maximum number of state POD modes")
    parser.add_argument("--max-m", type=int, default=1000, help="Maximum number of nonlinear POD/DEIM modes")
    parser.add_argument("--outdir", type=str, default=str(RESULTS_DIR), help="Output directory for plots and arrays")
    args = parser.parse_args()
    n = args.n
    epsilon = EPSILON
    b = FHN_B
    gamma = GAMMA
    c = C
    print("=" * 72)
    print("FitzHugh-Nagumo POD-DEIM reconstruction diagnostics")
    print("=" * 72)
    print(f"n = {n}, tf = {args.tf}, snapshots = {args.num_snapshots}")
    t0 = time.perf_counter()
    x, t, Y, G, A = solve_full_model(
        n=n,
        tf=args.tf,
        num_snapshots=args.num_snapshots,
        epsilon=epsilon,
        b=b,
        gamma=gamma,
        c=c,
    )
    full_time = time.perf_counter() - t0
    print(f"Full model solved in {full_time:.3f} s")
    red = build_reduced_operators(
        A=A,
        Y=Y,
        G=G,
        n=n,
        epsilon=epsilon,
        b=b,
        gamma=gamma,
        c=c,
        state_tol=args.state_tol,
        nonlin_tol=args.nonlin_tol,
        min_state_modes=args.min_k,
        min_nonlin_modes=args.min_m,
        max_state_modes=args.max_k,
        max_nonlin_modes=args.max_m,
    )
    print(f"Selected state POD dimension k = {red['k']}")
    print(f"Selected nonlinear POD/DEIM dimension m = {red['m']}")
    print(f"cond(P^T U_m) = {red['cond_ptu']:.3e}")
    v0, w0 = initial_condition(x)
    y0 = np.concatenate([v0, w0])
    y0_reduced = red["V_k"].T @ y0
    _, h, _ = build_neumann_laplacian_1d(n)
    g_small = boundary_input_vector(n)
    g_full = np.zeros(2 * n)
    g_full[:n] = g_small
    t1 = time.perf_counter()
    t_red, Y_red = solve_reduced_model(
        y0_reduced=y0_reduced,
        Ar=red["A_r"],
        B=red["B_deim"],
        V_k=red["V_k"],
        indices=red["indices"],
        r_const=red["r_const"],
        g=g_full,
        h=h,
        n=n,
        epsilon=epsilon,
        t0=0.0,
        tf=args.tf,
        num_snapshots=args.num_snapshots,
    )
    red_time = time.perf_counter() - t1
    Y_rom = red["V_k"] @ Y_red
    rel_err_total = relative_error(Y, Y_rom)
    rel_err_v = relative_error(Y[:n, :], Y_rom[:n, :])
    rel_err_w = relative_error(Y[n:, :], Y_rom[n:, :])
    timewise_err = pointwise_relative_errors(Y, Y_rom)
    print(f"Reduced model solved in {red_time:.3f} s")
    print(f"Speedup (solve only): {full_time / max(red_time, 1e-12):.2f}x")
    print(f"Relative error, full state: {rel_err_total:.6e}")
    print(f"Relative error, v block:    {rel_err_v:.6e}")
    print(f"Relative error, w block:    {rel_err_w:.6e}")
    print(f"Mean time-slice rel. err.:  {timewise_err.mean():.6e}")
    print(f"Max time-slice rel. err.:   {timewise_err.max():.6e}")
    outdir = Path(args.outdir)
    make_plots(
        outdir=outdir,
        x=x,
        t=t,
        Y_full=Y,
        Y_rom=Y_rom,
        n=n,
        svals_v=red["state_svals_v"],
        svals_w=red["state_svals_w"],
        nonlin_svals_v=red["nonlin_svals_v"],
        nonlin_svals_w=red["nonlin_svals_w"],
        node_index=n // 2,
    )
    np.savez(
        outdir / "results.npz",
        x=x,
        t=t,
        Y_full=Y,
        Y_rom=Y_rom,
        timewise_error=timewise_err,
        state_svals=red["state_svals"],
        nonlin_svals=red["nonlin_svals"],
        state_svals_v=red["state_svals_v"],
        state_svals_w=red["state_svals_w"],
        nonlin_svals_v=red["nonlin_svals_v"],
        nonlin_svals_w=red["nonlin_svals_w"],
        indices=red["indices"],
        k=red["k"],
        m=red["m"],
        rel_err_total=rel_err_total,
        rel_err_v=rel_err_v,
        rel_err_w=rel_err_w,
        full_time=full_time,
        red_time=red_time,
    )
    print(f"Saved plots and data in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
