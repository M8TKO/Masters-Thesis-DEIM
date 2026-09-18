import argparse
from pathlib import Path

import basix.ufl
import gmsh
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem import petsc
from dolfinx.io import gmsh as gmshio
from dolfinx.nls import petsc as nls
from mpi4py import MPI


INFLOW = 1
OUTFLOW = 2
WALLS = 3
CYLINDER = 5
FLUID = 10


def create_dfg_mesh(comm, n_bulk=64, n_circle=64, wake_refinement=0.4):
    """Create the DFG channel-with-cylinder mesh and boundary tags."""

    rank = 0
    if comm.rank == rank:
        gmsh.initialize()
        gmsh.model.add("dfg-2d")

        length = 2.2
        height = 0.41
        center = np.array([0.2, 0.2, 0.0])
        radius = 0.05

        rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, length, height)
        disk = gmsh.model.occ.addDisk(center[0], center[1], 0.0, radius, radius)
        fluid_entities, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, disk)])
        gmsh.model.occ.synchronize()

        fluid_surfaces = [tag for dim, tag in fluid_entities if dim == 2]
        gmsh.model.addPhysicalGroup(2, fluid_surfaces, FLUID)
        gmsh.model.setPhysicalName(2, FLUID, "fluid")

        boundary_entities = gmsh.model.getBoundary(
            [(2, tag) for tag in fluid_surfaces], oriented=False, recursive=False
        )

        inflow = []
        outflow = []
        walls = []
        cylinder = []

        for dim, tag in boundary_entities:
            x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(x, 0.0):
                inflow.append(tag)
            elif np.isclose(x, length):
                outflow.append(tag)
            elif np.isclose(y, 0.0) or np.isclose(y, height):
                walls.append(tag)
            else:
                cylinder.append(tag)

        if not cylinder:
            raise RuntimeError("Could not identify the cylinder boundary.")

        for marker, name, entities in [
            (INFLOW, "inflow", inflow),
            (OUTFLOW, "outflow", outflow),
            (WALLS, "walls", walls),
            (CYLINDER, "cylinder", cylinder),
        ]:
            gmsh.model.addPhysicalGroup(1, entities, marker)
            gmsh.model.setPhysicalName(1, marker, name)

        h_bulk = max(length, height) / n_bulk
        h_circle = 2.0 * np.pi * radius / n_circle
        h_wake = h_bulk * wake_refinement
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_bulk, h_circle))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_bulk)

        cylinder_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(cylinder_field, "CurvesList", cylinder)
        gmsh.model.mesh.field.setNumber(cylinder_field, "Sampling", 100)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", cylinder_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", h_circle)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", h_bulk)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", radius)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 4.0 * radius)

        wake_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(wake_field, "VIn", h_wake)
        gmsh.model.mesh.field.setNumber(wake_field, "VOut", h_bulk)
        gmsh.model.mesh.field.setNumber(wake_field, "XMin", center[0] + radius)
        gmsh.model.mesh.field.setNumber(wake_field, "XMax", length)
        gmsh.model.mesh.field.setNumber(wake_field, "YMin", 0.05)
        gmsh.model.mesh.field.setNumber(wake_field, "YMax", height - 0.05)

        mesh_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            mesh_field, "FieldsList", [threshold_field, wake_field]
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(mesh_field)

        gmsh.model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)

    if comm.rank == rank:
        gmsh.finalize()

    return mesh_data.mesh, mesh_data.facet_tags


def build_function_space(mesh):
    cell = mesh.basix_cell()
    velocity = basix.ufl.element("Lagrange", cell, 2, shape=(mesh.geometry.dim,))
    pressure = basix.ufl.element("Lagrange", cell, 1)
    mixed = basix.ufl.mixed_element([velocity, pressure])
    return fem.functionspace(mesh, mixed)


def ramped_amplitude(t, u_max, ramp_time):
    if ramp_time <= 0.0:
        return u_max
    return u_max * np.sin(0.5 * np.pi * min(t / ramp_time, 1.0))


def inlet_profile_values(x, amplitude, vertical_kick=0.0):
    height = 0.41
    values = np.zeros((2, x.shape[1]), dtype=x.dtype)
    values[0] = 4.0 * amplitude * x[1] * (height - x[1]) / (height**2)
    values[1] = vertical_kick * np.sin(np.pi * x[1] / height)
    return values


def startup_kick(t, u_max, strength, until, period):
    if strength == 0.0 or until <= 0.0 or t > until:
        return 0.0
    envelope = 0.5 * (1.0 + np.cos(np.pi * t / until))
    return strength * u_max * envelope * np.sin(2.0 * np.pi * t / period)


def inflow_velocity(V, amplitude):
    u_in = fem.Function(V)

    def profile(x):
        return inlet_profile_values(x, amplitude)

    u_in.interpolate(profile)
    return u_in


def update_inflow_velocity(u_in, amplitude, vertical_kick=0.0):
    def profile(x):
        return inlet_profile_values(x, amplitude, vertical_kick)

    u_in.interpolate(profile)
    u_in.x.scatter_forward()


def apply_initial_velocity(W, w, u_max, perturbation):
    V, velocity_dofs = W.sub(0).collapse()
    u0 = fem.Function(V)

    def profile(x):
        values = np.zeros((2, x.shape[1]), dtype=x.dtype)
        values[0] = 4.0 * u_max * x[1] * (0.41 - x[1]) / (0.41**2)
        values[1] = perturbation * np.sin(2.0 * np.pi * x[0] / 2.2) * np.sin(
            np.pi * x[1] / 0.41
        )
        return values

    u0.interpolate(profile)
    w.x.array[velocity_dofs] = u0.x.array
    w.x.scatter_forward()


def build_boundary_conditions(W, facet_tags, amplitude):
    mesh = W.mesh
    fdim = mesh.topology.dim - 1
    V, _ = W.sub(0).collapse()

    zero = fem.Function(V)
    inlet = inflow_velocity(V, amplitude)

    bcs = []
    for marker, value in [(WALLS, zero), (CYLINDER, zero), (INFLOW, inlet)]:
        facets = facet_tags.find(marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bcs.append(fem.dirichletbc(value, dofs, W.sub(0)))

    return bcs, inlet


def solve_unsteady_navier_stokes(
    W,
    facet_tags,
    *,
    nu=0.001,
    amplitude=1.0,
    t_end=8.0,
    dt=0.2,
    theta=0.5,
    ramp_time=2.0,
    perturbation=1.0e-3,
    newton_max_it=50,
    relaxation=1.0,
    save_every=5,
    min_dt=0.0025,
    max_retries=6,
    convection="picard",
    kick_strength=0.02,
    kick_until=4.0,
    kick_period=0.5,
    output="velocity_unsteady_navier_stokes.xdmf",
):
    bcs, inlet = build_boundary_conditions(W, facet_tags, 0.0)

    w = fem.Function(W, name="w")
    w_old = fem.Function(W, name="w_old")
    apply_initial_velocity(W, w, ramped_amplitude(0.0, amplitude, ramp_time), perturbation)
    w_old.x.array[:] = w.x.array
    w_old.x.scatter_forward()

    u, p = ufl.split(w)
    u_old, _ = ufl.split(w_old)
    v, q = ufl.TestFunctions(W)

    k = fem.Constant(W.mesh, dt)
    viscosity = fem.Constant(W.mesh, nu)
    theta_c = fem.Constant(W.mesh, theta)

    u_theta = theta_c * u + (1.0 - theta_c) * u_old
    if convection == "newton":
        convection_term = ufl.inner(ufl.grad(u_theta) * u_theta, v)
    elif convection == "picard":
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
    solver.atol = 1.0e-8
    solver.rtol = 1.0e-8
    solver.max_it = newton_max_it
    solver.relaxation_parameter = relaxation
    solver.error_on_nonconvergence = False

    ksp = solver.krylov_solver
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    with io.XDMFFile(W.mesh.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(W.mesh)

        t = 0.0
        step = 0
        update_inflow_velocity(
            inlet,
            ramped_amplitude(t, amplitude, ramp_time),
            startup_kick(t, amplitude, kick_strength, kick_until, kick_period),
        )

        while t < t_end - 1.0e-12:
            w_old.x.array[:] = w.x.array
            accepted = False
            retries = 0

            while not accepted:
                current_dt = min(dt, t_end - t)
                next_t = t + current_dt
                k.value = current_dt
                update_inflow_velocity(
                    inlet,
                    ramped_amplitude(next_t, amplitude, ramp_time),
                    startup_kick(
                        next_t, amplitude, kick_strength, kick_until, kick_period
                    ),
                )

                w.x.array[:] = w_old.x.array
                w.x.scatter_forward()
                iterations, converged = solver.solve(w)

                if converged:
                    accepted = True
                    continue

                retries += 1
                dt *= 0.5
                if W.mesh.comm.rank == 0:
                    print(
                        f"Newton failed at proposed t={next_t:.6g}; "
                        f"retrying with dt={dt:.6g}"
                    )

                if retries > max_retries or dt < min_dt:
                    raise RuntimeError(
                        "Newton solver did not converge after adaptive retries at "
                        f"step={step + 1}, t={next_t:.6g}. Last tried dt={dt:.6g}. "
                        "Try a smaller initial --dt, larger --ramp-time, lower "
                        "--relaxation, or coarser mesh for preview runs."
                    )

            w.x.scatter_forward()
            t = next_t
            step += 1

            u_high.interpolate(w.sub(0))
            u_out.interpolate(u_high)
            velocity_block_size = V_out.dofmap.index_map_bs
            local_max_vertical_velocity = np.max(
                np.abs(u_out.x.array.reshape((-1, velocity_block_size))[:, 1])
            )
            max_vertical_velocity = W.mesh.comm.allreduce(
                local_max_vertical_velocity, op=MPI.MAX
            )

            vorticity.interpolate(vorticity_expression)
            should_save = step % save_every == 0 or t >= t_end - 1.0e-12
            if should_save:
                xdmf.write_function(u_out, t)
                xdmf.write_function(vorticity, t)

            local_max_vorticity = np.max(np.abs(vorticity.x.array))
            max_vorticity = W.mesh.comm.allreduce(local_max_vorticity, op=MPI.MAX)

            if W.mesh.comm.rank == 0:
                print(
                    f"step={step:04d}, t={t:.6g}, "
                    f"newton_iterations={iterations}, "
                    f"dt={current_dt:.3g}, "
                    f"max|u_y|={max_vertical_velocity:.3e}, "
                    f"max|omega|={max_vorticity:.3e}, "
                    f"saved={should_save}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Unsteady DFG Navier-Stokes benchmark using DOLFINx/FEniCSx."
    )
    parser.add_argument("--n-bulk", type=int, default=64)
    parser.add_argument("--n-circle", type=int, default=64)
    parser.add_argument("--wake-refinement", type=float, default=0.4)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--u-max", type=float, default=1.5)
    parser.add_argument("--t-end", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--ramp-time", type=float, default=2.0)
    parser.add_argument("--perturbation", type=float, default=1.0e-3)
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--min-dt", type=float, default=0.0025)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--convection", choices=["picard", "newton"], default="picard"
    )
    parser.add_argument("--kick-strength", type=float, default=0.02)
    parser.add_argument("--kick-until", type=float, default=4.0)
    parser.add_argument("--kick-period", type=float, default=0.5)
    parser.add_argument(
        "--output", default="velocity_unsteady_navier_stokes.xdmf"
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    mesh, facet_tags = create_dfg_mesh(
        comm, args.n_bulk, args.n_circle, args.wake_refinement
    )
    W = build_function_space(mesh)
    solve_unsteady_navier_stokes(
        W,
        facet_tags,
        nu=args.nu,
        amplitude=args.u_max,
        t_end=args.t_end,
        dt=args.dt,
        theta=args.theta,
        ramp_time=args.ramp_time,
        perturbation=args.perturbation,
        newton_max_it=args.newton_max_it,
        relaxation=args.relaxation,
        save_every=args.save_every,
        min_dt=args.min_dt,
        max_retries=args.max_retries,
        convection=args.convection,
        kick_strength=args.kick_strength,
        kick_until=args.kick_until,
        kick_period=args.kick_period,
        output=args.output,
    )


if __name__ == "__main__":
    main()
