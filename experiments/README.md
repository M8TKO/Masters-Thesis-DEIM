# Numerical experiments

This directory contains the curated scripts used for the numerical examples in
the thesis. Generated arrays are excluded; each script writes its results locally.

After installing `requirements.txt` from the repository root, run:

```bash
python experiments/deim-1d/deim_1d.py
python experiments/deim-qdeim-2d/deim_qdeim_2d.py
python experiments/deim-qdeim-2d/restricted_qdeim.py
python experiments/svd-qr/svd_image_compression.py
python experiments/svd-qr/qr_image_compression.py
python experiments/svd-qr/svd_qr_error_comparison.py
python experiments/fitzhugh-nagumo/experiments/run_reconstruction.py
python experiments/fitzhugh-nagumo/experiments/make_figures.py
python experiments/rc-ladder/rc_ladder_experiment.py
```

The FitzHugh--Nagumo figure script reads the output produced by the preceding
reconstruction command.

The Navier--Stokes experiment has additional dependencies and external datasets;
see [navier-stokes/README.md](navier-stokes/README.md).
