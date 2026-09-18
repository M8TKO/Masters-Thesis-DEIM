# Data availability

Large generated simulation data is intentionally excluded from Git. The
`.gitignore` file prevents common snapshot and finite-element formats from being
committed accidentally.

## Cylinder-wake training data

The Navier--Stokes reconstruction experiment expects a dataset root containing
31 trajectories for

```text
u_max = 0.70, 0.71, ..., 1.00
```

Each trajectory contains 400 saved vorticity snapshots. The stored vorticity
matrix has shape `11994 x 400`, with time snapshots as columns.

The external validation trajectory uses the unseen value `u_max = 0.855` and has
400 snapshots. Its mesh has 11995 vorticity degrees of freedom; the reconstruction
script maps it to the training grid as documented in
`experiments/navier-stokes/reconstruction/methodology.md`.

Generate the parameter sweep with:

```bash
cd experiments/navier-stokes/solver
python run_u_max_sweep.py --output-root /path/to/dataset_u_max_sweep_dense
```

Generate the external trajectory by passing a one-value parameter range to the
same solver. Use absolute dataset paths when running the reconstruction scripts.

The compact CSV/JSON summaries required to inspect the reported results are
included in `experiments/navier-stokes/reconstruction/results/`.
