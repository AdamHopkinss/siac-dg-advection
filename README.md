# SIAC Postprocessing for DG Advection



This project investigates SIAC postprocessing applied to discontinuous Galerkin approximations of the one-dimensional linear advection equation.



The initial goals are:



- implement a modal DG solver for 1D linear advection,

- initialize the solution using an L2 projection,

- evolve the DG solution using RK4 time stepping,

- add controlled Gaussian noise to the initial data,

- apply symmetric and boundary-aware SIAC filters,

- compare filtered and unfiltered DG solutions.



## Project structure



```text

src/        Core implementation

scripts/    Reproducible experiments

notebooks/  Exploratory analysis

results/    Numerical data

figures/    Generated plots

tests/      Unit tests

