# lineint

Class to compute the segments of a line on a grid.

## Purpose and motivation

When integrating along a line on a grid, one can use this code to determine the size of the segments of the line contained within each grid cell. One can then use these segments as weighting terms in a numerical integration. While this code can be applied to various scenarios, the motivation was initially for numerical relativity where each grid position has a different volume element. So to have the proper length between two grid positions, one needs to numerically integrate with the determinant of the spatial metric, see Appendix A of reference paper.

## Install

Download the 'lineint.py' file and place it in your working directory. You can then call it in your script or notebook with 'import lineint'.

## Defining the class

This file only has the class **LineIntegrate** inside and it needs to be initialised with the x, y and z grid domains and the indices of the starting point of the line. This is done by passing four lists: three that each contain the minimum and maximum values and grid spacing in each direction, and one list of integers for the starting point indices. **li = lineint.LineIntegrate(xdomain, ydomain, zdomain, starting_point)**

The class will then define an 'input grid' based on the passed domains. This will be embodied with three arrays: **li.igridx**, **li.igridy**, **li.igridz**. It will also define a 'grid around zero', with the same size and grid spacing, but the (0.0, 0.0, 0.0) coordinate poisition will be placed in the midle of the grid. This will be embodied with three arrays: **li.grid0x**, **li.grid0y**, **li.grid0z**.

By construction, the data on the input grid is assumed to be periodic, and the purpose of the secondary grid, around zero, is to shift the starting point to be at the (0.0, 0.0, 0.0) location. This significantly simplifies computations as it avoids dealing with boundaries, but it also limits this code to calculate segments of lines that are half the box size.

Note that, indices and data are shifted from one grid to the other with the functions: **li.idx_igrid_to_grid0**, **li.idx_grid0_to_igrid** and **li.shift_igrid_to_grid0**.

  

## Using this code

There are three scenarios where this code can be used, these are each presented in the Example notebook and their accuracy is presented in the Tests notebook. The three cases are:

**1.** from the starting point indice to a chosen end point indice, use the function:
**indices_igrid, segments, theta, phi = li.segments_idx_end(end_point)**

**2.** from the starting point indice to a chosen radius in a given angular direction:
the variable **li.maximum_radius** needs to be redefined, then use the function
**indices_igrid, segments = li.segments_radius_end(phi, theta)**

**3.** from (0.0, 0.0, 0.0) to a sign change in some data in a given angular direction.
I refer to this data as a condition, and because this sign change can be between grid positions, a function of the condition needs to be provided, either the function itself when available, or an interpolation function.
**indices_grid0, segments = li.segments_conditional_end(phi, theta, condition_grid0, condition_function_grid0)**
Note that this function assumes the starting point to be at (0.0, 0.0, 0.0) so make sure that condition and condition_function are centered around zero. The Example notebook shows how to shift the data for this.

Note that these three functions wrap around the main function:
**indices_grid0, segments, radius, coord = li.idx_along_r(rend, phi, theta)**

  

## Reference

If you use this code please reference

@article{R.L.Munoz_M.Bruni_2023,
title = {Structure formation and quasispherical collapse from initial curvature perturbations with numerical relativity simulations},
author  = {Munoz, R. L. and Bruni, M.},
journal = {Physical Review D},
volume  = {107},
number  = {12},
pages = {123536},
numpages  = {26},
year  = {2023},
month = {6},
doi = {10.1103/PhysRevD.107.123536},
archivePrefix = {arXiv},
eprint  = {astro-ph/2302.09033}}
