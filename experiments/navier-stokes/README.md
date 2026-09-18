# Parametric cylinder-wake experiment

This experiment solves two-dimensional incompressible flow past a circular
obstacle for a parameter sweep of maximum inlet velocities, stores vorticity
snapshots, and compares POD projection, DEIM and QDEIM reconstruction.

It is a sparse field-reconstruction experiment, not a hyper-reduced
Navier--Stokes solver.

## Structure

```text
solver/          FEniCSx/DOLFINx full-order solver and dataset generator
reconstruction/  POD, DEIM and QDEIM analysis and compact reported results
```

## Environment

Create the repository environment with:

```bash
conda env create -f environment.yml
conda activate masters-thesis-deim
```

## Generate snapshots

```bash
cd experiments/navier-stokes/solver
python run_u_max_sweep.py --output-root /path/to/dataset_u_max_sweep_dense
```

The generated data is large and is ignored by Git. See the repository-level
`DATA.md` for the expected layout.

## Reproduce the reconstruction study

```bash
python experiments/navier-stokes/reconstruction/scripts/run_experiments.py \
  --dataset-root /path/to/dataset_u_max_sweep_dense \
  --validation-root /path/to/validation_u_max_0_855 \
  --output-root /path/to/output \
  --selected-m 150

python experiments/navier-stokes/reconstruction/scripts/make_figures.py \
  --dataset-root /path/to/dataset_u_max_sweep_dense \
  --output-root /path/to/output
```

The repository includes the compact CSV/JSON results and rendered PNG figures,
but excludes bases, full fields, ParaView files and raw snapshot matrices.

## Attribution

The geometry, benchmark setting and parts of the workflow were developed with
reference to Jørgen S. Dokken's FEniCSx tutorial example, *Test problem 2: Flow
past a cylinder (DFG 2D-3 benchmark)*:

https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

That tutorial is licensed under CC BY 4.0. This implementation changes the
discretization and workflow to support a parametric inlet-velocity sweep,
vorticity snapshot export and DEIM/QDEIM reconstruction. See
`THIRD_PARTY_NOTICES.md` at the repository root.
