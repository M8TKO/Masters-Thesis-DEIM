import argparse
import json
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem import petsc
from dolfinx.nls import petsc as nls
from mpi4py import MPI

from dfg import (
    build_boundary_conditions,
    build_function_space,
    create_dfg_mesh,
    ramped_amplitude,
    startup_kick,
    update_inflow_velocity,
    apply_initial_velocity,
)


def parameter_label(value):
    return f"{value:.3f}".replace(".", "_")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_solver(W, facet_tags, params, u_max):
    bcs, inlet = build_boundary_conditions(W, facet_tags, 0.0)

    w = fem.Function(W, name="w")
    w_old = fem.Function(W, name="w_old")
    apply_initial_velocity(
        W,
        w,
        ramped_amplitude(0.0, u_max, params["ramp_time"]),
        params["perturbation"],
    )
    w_old.x.array[:] = w.x.array
    w_old.x.scatter_forward()

    u, p = ufl.split(w)
    u_old, _ = ufl.split(w_old)
    v, q = ufl.TestFunctions(W)

    k = fem.Constant(W.mesh, params["dt"])
    viscosity = fem.Constant(W.mesh, params["nu"])
    theta_c = fem.Constant(W.mesh, params["theta"])

    u_theta = theta_c * u + (1.0 - theta_c) * u_old
    if params["convection"] == "newton":
        convection_term = ufl.inner(ufl.grad(u_theta) * u_theta, v)
    elif params["convection"] == "picard":
        convection_term = ufl.inner(ufl.grad(u_theta) * u_old, v)
    else:
        raise ValueError("convection must be 'picard' or 'newton'")

    F = (
        ufl.inner((u - u_old) / k, v)
        + viscosity * ufl.inner(ufl.grad(u_theta), ufl.grad(v))
        + convection_term
        - p * ufl.div(v)
        - q * ufl.div(u)
    ) * ufl.dx
    J = ufl.derivative(F, w)

    problem = petsc.NewtonSolverNonlinearProblem(F, w, bcs=bcs, J=J)
    solver = nls.NewtonSolver(W.mesh.comm, problem)
    solver.atol = params["newton_atol"]
    solver.rtol = params["newton_rtol"]
    solver.max_it = params["newton_max_it"]
    solver.relaxation_parameter = params["relaxation"]
    solver.error_on_nonconvergence = False

    ksp = solver.krylov_solver
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass

    return solver, w, w_old, inlet, k


def solve_and_collect_vorticity(W, facet_tags, params, u_max, run_dir):
    solver, w, w_old, inlet, k = build_solver(W, facet_tags, params, u_max)

    velocity_out_element = basix.ufl.element(
        "Lagrange", W.mesh.basix_cell(), 1, shape=(W.mesh.geometry.dim,)
    )
    V_high, _ = W.sub(0).collapse()
    u_high = fem.Function(V_high, name="velocity_high_order")
    V_out = fem.functionspace(W.mesh, velocity_out_element)
    u_out = fem.Function(V_out, name="velocity")

    vorticity_element = basix.ufl.element("Lagrange", W.mesh.basix_cell(), 1)
    V_vorticity = fem.functionspace(W.mesh, vorticity_element)
    vorticity = fem.Function(V_vorticity, name="vorticity")
    vorticity_expression = fem.Expression(
        u_high[1].dx(0) - u_high[0].dx(1),
        V_vorticity.element.interpolation_points,
    )

    vorticity_snapshots = []
    times = []

    label = parameter_label(u_max)
    output_path = run_dir / f"wake_u_max_{label}.xdmf"

    with io.XDMFFile(W.mesh.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(W.mesh)

        t = 0.0
        step = 0
        update_inflow_velocity(
            inlet,
            ramped_amplitude(t, u_max, params["ramp_time"]),
            startup_kick(
                t,
                u_max,
                params["kick_strength"],
                params["kick_until"],
                params["kick_period"],
            ),
        )

        while t < params["t_end"] - 1.0e-12:
            w_old.x.array[:] = w.x.array
            accepted = False
            retries = 0

            while not accepted:
                current_dt = min(params["dt"], params["t_end"] - t)
                next_t = t + current_dt
                k.value = current_dt
                update_inflow_velocity(
                    inlet,
                    ramped_amplitude(next_t, u_max, params["ramp_time"]),
                    startup_kick(
                        next_t,
                        u_max,
                        params["kick_strength"],
                        params["kick_until"],
                        params["kick_period"],
                    ),
                )

                w.x.array[:] = w_old.x.array
                w.x.scatter_forward()
                iterations, converged = solver.solve(w)

                if converged:
                    accepted = True
                    continue

                retries += 1
                params["dt"] *= 0.5
                print(
                    f"u_max={u_max:.3f}: Newton failed at proposed "
                    f"t={next_t:.6g}; retrying with dt={params['dt']:.6g}",
                    flush=True,
                )

                if retries > params["max_retries"] or params["dt"] < params["min_dt"]:
                    raise RuntimeError(
                        "Newton solver did not converge after adaptive retries for "
                        f"u_max={u_max:.3f}, step={step + 1}, t={next_t:.6g}. "
                        f"Last tried dt={params['dt']:.6g}."
                    )

            w.x.scatter_forward()
            t = next_t
            step += 1

            u_high.interpolate(w.sub(0))
            u_out.interpolate(u_high)
            vorticity.interpolate(vorticity_expression)

            should_save = step % params["save_every"] == 0 or t >= params["t_end"] - 1.0e-12
            if should_save:
                xdmf.write_function(u_out, t)
                xdmf.write_function(vorticity, t)
                times.append(t)
                vorticity_snapshots.append(vorticity.x.array.copy())

            max_vorticity = np.max(np.abs(vorticity.x.array))
            print(
                f"u_max={u_max:.3f}, step={step:04d}, t={t:.6g}, "
                f"newton_iterations={iterations}, max|omega|={max_vorticity:.3e}, "
                f"saved={should_save}",
                flush=True,
            )

    vorticity_matrix = np.column_stack(vorticity_snapshots)
    times_array = np.array(times)
    coordinates = V_vorticity.tabulate_dof_coordinates()

    np.save(run_dir / "vorticity_matrix.npy", vorticity_matrix)
    np.save(run_dir / "times.npy", times_array)
    np.save(run_dir / "coordinates.npy", coordinates)

    run_metadata = {
        **params,
        "u_max": u_max,
        "matrix_file": "vorticity_matrix.npy",
        "times_file": "times.npy",
        "coordinates_file": "coordinates.npy",
        "xdmf_file": output_path.name,
        "h5_file": output_path.with_suffix(".h5").name,
        "matrix_shape": list(vorticity_matrix.shape),
        "matrix_orientation": "columns_are_time_snapshots",
        "field": "vorticity",
        "field_element_family": "Lagrange",
        "field_element_degree": 1,
    }
    write_json(run_dir / "parameters.json", run_metadata)

    return run_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vorticity snapshot matrices for a uniform u_max sweep."
    )
    parser.add_argument("--output-root", default="dataset_u_max_sweep")
    parser.add_argument("--u-min", type=float, default=0.7)
    parser.add_argument("--u-max", type=float, default=1.0)
    parser.add_argument("--num-params", type=int, default=7)
    parser.add_argument("--n-bulk", type=int, default=56)
    parser.add_argument("--n-circle", type=int, default=112)
    parser.add_argument("--wake-refinement", type=float, default=0.25)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--t-end", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--ramp-time", type=float, default=4.0)
    parser.add_argument("--perturbation", type=float, default=1.0e-2)
    parser.add_argument("--kick-strength", type=float, default=0.04)
    parser.add_argument("--kick-until", type=float, default=6.0)
    parser.add_argument("--kick-period", type=float, default=0.6)
    parser.add_argument("--newton-atol", type=float, default=1.0e-8)
    parser.add_argument("--newton-rtol", type=float, default=1.0e-8)
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--min-dt", type=float, default=0.0025)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--convection", choices=["picard", "newton"], default="picard"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("run_u_max_sweep.py is intended for serial execution.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    u_values = np.linspace(args.u_min, args.u_max, args.num_params)

    base_params = {
        "n_bulk": args.n_bulk,
        "n_circle": args.n_circle,
        "wake_refinement": args.wake_refinement,
        "nu": args.nu,
        "t_end": args.t_end,
        "dt": args.dt,
        "save_every": args.save_every,
        "theta": args.theta,
        "ramp_time": args.ramp_time,
        "perturbation": args.perturbation,
        "kick_strength": args.kick_strength,
        "kick_until": args.kick_until,
        "kick_period": args.kick_period,
        "newton_atol": args.newton_atol,
        "newton_rtol": args.newton_rtol,
        "newton_max_it": args.newton_max_it,
        "relaxation": args.relaxation,
        "min_dt": args.min_dt,
        "max_retries": args.max_retries,
        "convection": args.convection,
    }

    print("Creating shared mesh...", flush=True)
    mesh, facet_tags = create_dfg_mesh(
        comm, args.n_bulk, args.n_circle, args.wake_refinement
    )
    W = build_function_space(mesh)

    with io.XDMFFile(comm, str(output_root / "mesh.xdmf"), "w") as mesh_file:
        mesh_file.write_mesh(mesh)

    run_summaries = []
    for index, u_value in enumerate(u_values, start=1):
        label = parameter_label(u_value)
        run_dir = output_root / f"u_max_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== Run {index}/{len(u_values)}: u_max={u_value:.3f} ===",
            flush=True,
        )
        params = dict(base_params)
        run_metadata = solve_and_collect_vorticity(
            W, facet_tags, params, float(u_value), run_dir
        )
        run_summaries.append(
            {
                "u_max": float(u_value),
                "directory": str(run_dir.relative_to(output_root)),
                "matrix_shape": run_metadata["matrix_shape"],
                "time_start": float(np.load(run_dir / "times.npy")[0]),
                "time_end": float(np.load(run_dir / "times.npy")[-1]),
            }
        )

    dataset_metadata = {
        "description": "DFG cylinder wake vorticity matrices for a uniform u_max sweep.",
        "u_values": [float(v) for v in u_values],
        "base_parameters": base_params,
        "mesh_file": "mesh.xdmf",
        "mesh_h5_file": "mesh.h5",
        "matrix_orientation": "columns_are_time_snapshots",
        "field": "vorticity",
        "field_element_family": "Lagrange",
        "field_element_degree": 1,
        "runs": run_summaries,
    }
    write_json(output_root / "metadata.json", dataset_metadata)

    print(f"\nDataset written to {output_root}", flush=True)


if __name__ == "__main__":
    main()
