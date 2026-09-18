# Third-party notices

## FEniCSx cylinder-flow tutorial

The geometry, benchmark setting and parts of the numerical workflow in
`experiments/navier-stokes/solver/` were developed with reference to the DFG
cylinder-flow example in the FEniCSx tutorial:

- Jørgen S. Dokken, *Test problem 2: Flow past a cylinder (DFG 2D-3 benchmark)*
- https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

The tutorial is distributed under the Creative Commons Attribution 4.0
International License. This repository reimplements and modifies the setup for a
parametric inlet-velocity sweep, monolithic Taylor--Hood discretization, vorticity
snapshot export and DEIM/QDEIM reconstruction. The original author does not
endorse this repository.

## Software dependencies

NumPy, SciPy, Matplotlib, Pillow, h5py, mpi4py, Gmsh and the FEniCSx/DOLFINx
components are external dependencies and are not redistributed here. Each remains
subject to its own license.

## Input photograph

`experiments/svd-qr/image.jpg` is an original photograph supplied by the thesis
author and is used only as numerical input for the image-compression examples. It
is not covered by the MIT software license and remains copyright Matko Petričić.
