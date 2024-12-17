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
