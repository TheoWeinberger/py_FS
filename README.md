# README for fs_plot.py

Copyright 2023 T I Weinberger

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Python script for plotting Fermi surfaces in .bxsf format.

## Usage
In a directory containing .bxsf files with format case.bxsf.band-n run fs_plot.py case to plot the fermi surface.

To plot in the conventional extended Brillouin zone, use:

```bash
python fs_plot.py -n ute2 --conventional-bz
```

This option requires `spglib` (`pip install spglib`). It determines the
conventional cell of the **reciprocal lattice** while preserving the input
Cartesian orientation, then draws its Wigner–Seitz boundary. The shape follows
the lattice: for example, a box for orthorhombic/tetragonal reciprocal cells and
a hexagonal prism for hexagonal cells. For the supplied UTe₂ files, the box spans
±0.12834799, ±0.08695000 and ±0.03831300 along the input Cartesian axes, in BXSF
reciprocal units.

Band energies and optional scalar fields are periodically sampled throughout
this zone, and the Fermi surface is clipped to the same boundary. This is an
extended-zone visualization; it does not compute spectral unfolding weights.
Without the flag, the original primitive Wigner–Seitz BZ is used.

## Dependencies

This code has been developed and tested with the following packages on a Linux 22.04 LTS system:
    - python 3.9.10
    - numpy 1.20.3
    - matplotlib 3.7.1
    - scipy 1.7.3
    - pyvista 0.37.0
    - triangle 1.6
    - pandas 1.5.3 
    - seaborn 0.11.2
    - copy from python 3.9.10

To use SKEAF for calculating extremal orbits the code expects to find an executable named skeaf
in the users bin. The correct version of SKEAF can be found in the folder fs_plot_skeaf along with a 
makefile for ifort openmp compilation

## Examples

Directory contains example .bxsf files foir UTe2. To test, run fs_plot.py UTe2
