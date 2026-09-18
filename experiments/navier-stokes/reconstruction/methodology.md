# Methodology

## Problem Setting

The full-order data comes from two-dimensional incompressible flow past a circular obstacle in a channel. The parameter is the maximum inlet velocity,

\[
\mu = U_{\max}.
\]

For each parameter value, the Navier-Stokes solver produces a time-dependent velocity field \(u(x,t;\mu)\). The scalar field used for model reduction is the vorticity,

\[
\omega(x,t;\mu)
=
\frac{\partial u_2}{\partial x}
-
\frac{\partial u_1}{\partial y}.
\]

The stored data matrices contain nodal vorticity values. Each column is one saved time snapshot.

## Training And Validation Separation

### Experiment 1

The first experiment is a leave-one-parameter-out test. The trajectory at

\[
\mu = 0.85
\]

is held out completely. The POD mean and basis are computed only from the remaining 30 parameter trajectories. Therefore,

\[
X_{\mathrm{train}}\in\mathbb R^{11994\times12000}.
\]

The held-out matrix has shape

\[
X_{\mathrm{test}}\in\mathbb R^{11994\times400}.
\]

The output file `results/mode_sweep.json` records the training parameter values and verifies that no \(\mu=0.85\) snapshots are used in the POD mean or basis.

### Experiment 2

The second experiment trains on all 31 dense-grid parameter trajectories,

\[
\mu = 0.70, 0.71, \ldots, 1.00,
\]

and validates at the external parameter

\[
\mu = 0.855.
\]

This value is not part of the training grid. The selected dimension is \(m=150\), chosen from Experiment 1 and not tuned on the external validation data.

## Centering

Let \(X_{\mathrm{train}}\) denote the training snapshot matrix. The mean snapshot is

\[
\bar\omega
=
\frac{1}{N}\sum_{j=1}^N X_{\mathrm{train},j}.
\]

The centered matrix is

\[
X_c
=
X_{\mathrm{train}}-\bar\omega\mathbf 1^T.
\]

All POD bases are computed from \(X_c\). Reconstructions add the mean back after reduced coefficients are computed.

## Randomized POD

A randomized SVD procedure is used to compute a basis up to the largest requested dimension. The random seed is fixed at `1234`.

For Experiment 1, one POD basis is computed up to \(m=200\). Smaller spaces are obtained by truncating the same basis:

\[
U_m = U_{200}[:,1:m].
\]

This ensures that all mode-sweep comparisons use nested POD spaces.

The orthonormality check

\[
\|U_m^T U_m-I\|_2
\]

is recorded for every \(m\).

The cumulative-energy figure uses the exact centered training-matrix energy as its denominator:

\[
\frac{\sum_{i=1}^k \sigma_i^2}{\|X_c\|_F^2}.
\]

The denominator and the energy fraction represented by the computed randomized singular values are stored in `results/singular_value_decay.json`.

## POD Projection

The POD projection is the best Euclidean reconstruction in the selected POD subspace when the complete field is known. For a test matrix \(X\), the projection is

\[
X_{\mathrm{POD}}
=
\bar\omega\mathbf 1^T
+
U_m U_m^T
(X-\bar\omega\mathbf 1^T).
\]

This is not a sparse measurement method. It is included as a lower reference error for the chosen POD space.

## Classical Greedy DEIM

Classical DEIM selects interpolation indices recursively from the POD basis vectors. The first index is the position of the largest absolute entry of the first basis vector. Subsequent indices are chosen as the largest absolute entry of the residual after interpolating the next basis vector through the previously selected indices.

Let \(P\in\mathbb R^{n\times m}\) be the selection matrix associated with the selected indices. The DEIM reconstruction of a centered field is obtained from the selected entries only:

\[
\omega_{\mathrm{DEIM}}
=
\bar\omega
+
U_m
(P^T U_m)^{-1}
P^T(\omega-\bar\omega).
\]

In the implementation, this is computed with a linear solve, not by forming an explicit inverse.

## QDEIM

QDEIM selects indices using column-pivoted QR applied to

\[
U_m^T.
\]

The first \(m\) QR pivots are used as the sensor indices. Reconstruction then uses the same interpolation formula:

\[
\omega_{\mathrm{QDEIM}}
=
\bar\omega
+
U_m
(P^T U_m)^{-1}
P^T(\omega-\bar\omega).
\]

Again, the coefficients are computed with linear solves.

## Error Definitions

Let \(X\) be the reference matrix and \(Y\) the reconstruction.

The relative Frobenius error is

\[
E_F
=
\frac{\|X-Y\|_F}{\|X\|_F}.
\]

For each time snapshot \(j\), the snapshot-relative Euclidean error is

\[
e_j
=
\frac{\|X_j-Y_j\|_2}{\|X_j\|_2}.
\]

The reported time-local statistics are:

- mean of \(e_j\);
- median of \(e_j\);
- 95th percentile of \(e_j\);
- maximum of \(e_j\).

For DEIM and QDEIM, the following interpolation-matrix quantities are also recorded:

\[
\operatorname{cond}_2(P^T U_m),
\]

and

\[
\|(P^T U_m)^{-1}\|_2.
\]

The implementation computes the second quantity by solving against the identity matrix.

## Verification Checks

The experiment records the following checks:

- POD orthonormality \(\|U_m^TU_m-I\|_2\);
- exactly \(m\) distinct DEIM/QDEIM indices;
- nonsingularity of \(P^TU_m\);
- interpolation residual at selected entries,

\[
P^T\omega_{\mathrm{rec}}-P^T\omega;
\]

- exclusion of \(\mu=0.85\) from the leave-one-out training basis;
- exclusion of \(\mu=0.855\) from the external-validation training grid;
- reuse of the same POD basis and mean for all methods at fixed \(m\).

## Mesh Alignment For External Validation

The external validation simulation at \(\mu=0.855\) has 11995 vorticity degrees of freedom. The dense training data has 11994. Therefore, the external vorticity snapshots are mapped to the training mesh before reconstruction.

The mesh transfer uses linear interpolation from the external mesh coordinates to the training mesh coordinates. If a target point is outside the interpolation simplex or produces a missing value, nearest-neighbour fallback is used.

The file `results/external_validation.json` records:

- source mesh size;
- target mesh size;
- interpolation method;
- number of fallback points;
- maximum fallback distance;
- round-trip interpolation control error.

The round-trip test maps the \(u_{\max}=0.70\) training trajectory to the external validation mesh and back to the training mesh. This estimates the magnitude of interpolation error introduced by the mesh transfer.

The largest external snapshot-relative errors occur at the first saved time, \(t=0.02\), where the norm of the vorticity field is small. The external field figures therefore emphasize developed wake times \(t=6.00\) and \(t=8.00\), while the early-time maximum is shown in the time-error curve.

Two vorticity visualizations are generated. The main external vorticity figure uses a common robust scale across the shown truth and reconstruction fields. The wake-contrast figure uses a narrower symmetric scale derived from the 99.5th percentile of \(|\omega|\) in the displayed wake region, excluding the immediate cylinder neighbourhood. This second figure is intended for visual inspection of wake structures; its scale is recorded in `results/wake_contrast_figure_scale.json`.

## Why This Is Not A Hyper-Reduced Navier-Stokes Solver

This experiment reconstructs already available vorticity states from sparse spatial samples. The time-dependent Navier-Stokes equations are not projected and evolved in a reduced space. The nonlinear Navier-Stokes operator is not approximated by DEIM inside a reduced PDE solver.

Therefore, the correct interpretation is sparse state reconstruction:

\[
\text{selected vorticity values}
\quad\longmapsto\quad
\text{full vorticity field approximation}.
\]

It should not be described as a complete reduced-order Navier-Stokes simulation or as a hyper-reduced Navier-Stokes solver.
