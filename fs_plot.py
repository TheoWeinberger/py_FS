#!/usr/bin/env python3
"""
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

Plot Fermi Surfaces
"""

import numpy as np
from matplotlib import pyplot as plt
import pyvista as pv
import seaborn as sns
import matplotlib as mpl
import re
import glob
import os
import subprocess
import scienceplots
import io
from scipy.ndimage import map_coordinates
import argparse
import ast
import pandas as pd

pv.global_theme.colorbar_orientation = "vertical"

# define some plotting parameters for rough
# plotting of output
plt.style.use(["nature"])
cp = sns.color_palette("Set2")
# pv.rcParams['transparent_background'] = True


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1
            / 2
            * (
                1
                - abs(
                    (self.midpoint - self.vmin) / (self.midpoint - self.vmax)
                )
            ),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (
                1
                + abs(
                    (self.vmax - self.midpoint) / (self.midpoint - self.vmin)
                )
            ),
        )
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


def read_bxsf_info(file_name):

    # open and read file
    f = open(file_name, "r")
    lines = f.readlines()

    """
    this corresponds to code for getting the fermi energy
    """
    # get fermi energy
    x = lines[1].split(" ")
    x = [val for _, val in enumerate(x) if val != ""]
    x = [val for _, val in enumerate(x) if val != "\t"]
    x = x[-1].split("\n")
    # convert ef to a number
    e_f = float(x[0])

    # get fermi energy
    band_index = re.findall(r"\d+", lines[12])[0]

    """
    Now get the dimensions of the k mesh
    """

    # get dimensions
    x = re.findall(r"\d+", lines[7])
    # convert dimensions to integer values
    dimensions = [int(dim) for dim in x]

    """
    Get basis vectors
    """
    # first spanning vector
    vec_1 = lines[9].split(" ")
    vec_1 = [val.strip("\n") for val in vec_1]
    vec_1 = [val.strip("\t") for val in vec_1]
    vec_1 = [val for _, val in enumerate(vec_1) if val != ""]
    # convert dimensions to float values
    vec_1 = [float(val) for val in vec_1]

    # second spanning vector
    vec_2 = lines[10].split(" ")
    vec_2 = [val.strip("\n") for val in vec_2]
    vec_2 = [val.strip("\t") for val in vec_2]
    vec_2 = [val for _, val in enumerate(vec_2) if val != ""]
    # convert dimensions to float values
    vec_2 = [float(val) for val in vec_2]

    # thrid spanning vector
    vec_3 = lines[11].split(" ")
    vec_3 = [val.strip("\n") for val in vec_3]
    vec_3 = [val.strip("\t") for val in vec_3]
    vec_3 = [val for _, val in enumerate(vec_3) if val != ""]
    # convert dimensions to float values
    vec_3 = [float(val) for val in vec_3]

    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    vec_3 = np.array(vec_3)

    cell = np.array([vec_1, vec_2, vec_3])

    # close file
    f.close()

    return vec_1, vec_2, vec_3, dimensions, band_index, e_f, cell


def read_bxsf(
    file_name, scale, order, shift_energy, fermi_velocity, scalar="None"
):
    """
    Reads .bxsf file and determines two
    matrices, one corresponding to the eigenvalues
    and the other the k space vectors

    returns:
        k_vectors
        eigenvalues
        ef
    """
    # open and read file
    f = open(file_name, "r")
    lines = f.readlines()

    """
    Get general bxsf information
    """
    vec_1, vec_2, vec_3, dimensions, band_index, e_f, cell = read_bxsf_info(
        file_name
    )

    vec_1 = vec_1 * (dimensions[0] + 1) / dimensions[0]
    vec_2 = vec_2 * (dimensions[1] + 1) / dimensions[1]
    vec_3 = vec_3 * (dimensions[2] + 1) / dimensions[2]

    """
    Now extract eigenvalues
    """

    # get everything after band index
    lines_conc = "".join(lines)
    for line in lines:
        if "BAND:" in line and band_index in line:
            eigen_text = lines_conc.split(line)[1]
            break
    eigen_text = eigen_text.split("END_BANDGRID_3D")[0]
    eigen_text = re.sub(" +", " ", eigen_text)
    eigen_vals = eigen_text.split(" ")
    eigen_vals = [val.strip("\n") for val in eigen_vals]
    eigen_vals = eigen_vals[1:-1]
    eigen_vals = [float(val) for val in eigen_vals]

    eigenvalues = np.array(eigen_vals).reshape(dimensions)

    eig_vals = np.zeros(dimensions)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                eig_vals[i, j, k] = eigen_vals[
                    dimensions[2] * dimensions[1] * i + dimensions[2] * j + k
                ]

    eig_vals = np.tile(eig_vals, (2, 2, 2))

    eig_vals = np.delete(eig_vals, dimensions[0], axis=0)
    eig_vals = np.delete(eig_vals, dimensions[1], axis=1)
    eig_vals = np.delete(eig_vals, dimensions[2], axis=2)

    if scale != 1:

        # interpolation here
        dimensions_int = [int(scale * dimension) for dimension in dimensions]
        x_vals_int = np.linspace(
            0, 2 * dimensions[0] - 2, 2 * dimensions_int[0] - 2 * scale + 1
        )
        y_vals_int = np.linspace(
            0, 2 * dimensions[1] - 2, 2 * dimensions_int[1] - 2 * scale + 1
        )
        z_vals_int = np.linspace(
            0, 2 * dimensions[2] - 2, 2 * dimensions_int[2] - 2 * scale + 1
        )

        x_vals_int, y_vals_int, z_vals_int = np.meshgrid(
            x_vals_int, y_vals_int, z_vals_int
        )

        out_grid_x, out_grid_y, out_grid_z = np.meshgrid(
            range(-dimensions_int[0] + scale, dimensions_int[0] - scale + 1),
            range(-dimensions_int[1] + scale, dimensions_int[1] - scale + 1),
            range(-dimensions_int[2] + scale, dimensions_int[2] - scale + 1),
        )

        out_grid_x = out_grid_x / scale
        out_grid_y = out_grid_y / scale
        out_grid_z = out_grid_z / scale

        out_coords = np.array(
            (x_vals_int.ravel(), y_vals_int.ravel(), z_vals_int.ravel())
        )

        # x_vals_int, y_vals_int, z_vals_int = np.meshgrid(x_vals_int, y_vals_int, z_vals_int)

        out_data = map_coordinates(
            eig_vals, out_coords, order=order, mode="reflect"
        )

        out_data = out_data.reshape(
            2 * dimensions_int[0] - 2 * scale + 1,
            2 * dimensions_int[1] - 2 * scale + 1,
            2 * dimensions_int[2] - 2 * scale + 1,
        )

    else:

        out_data = eig_vals
        dimensions_int = dimensions
        out_grid_y, out_grid_x, out_grid_z = np.meshgrid(
            range(-dimensions_int[0] + 1, dimensions_int[0]),
            range(-dimensions_int[1] + 1, dimensions_int[1]),
            range(-dimensions_int[2] + 1, dimensions_int[2]),
        )

    x_vals = []
    y_vals = []
    z_vals = []

    # Transform index matrix into coordinate matrix in k space
    k_vectors = np.zeros(
        [2 * dimension + 1 - 2 * scale for dimension in dimensions_int] + [3]
    )

    k_vectors[:] = (
        np.outer(out_grid_x, vec_1).reshape(
            [2 * dimension + 1 - 2 * scale for dimension in dimensions_int]
            + [3]
        )
        + np.outer(out_grid_y, vec_2).reshape(
            [2 * dimension + 1 - 2 * scale for dimension in dimensions_int]
            + [3]
        )
        + np.outer(out_grid_z, vec_3).reshape(
            [2 * dimension + 1 - 2 * scale for dimension in dimensions_int]
            + [3]
        )
    )

    k_vectors = k_vectors.reshape(
        (2 * dimensions_int[0] + 1 - 2 * scale)
        * (2 * dimensions_int[1] + 1 - 2 * scale)
        * (2 * dimensions_int[2] + 1 - 2 * scale),
        3,
    )
    x_vals = k_vectors[:, 0]
    y_vals = k_vectors[:, 1]
    z_vals = k_vectors[:, 2]

    # create a structured grid to store the data and reshape to the correct dimensions
    grid = pv.StructuredGrid(
        np.array(x_vals) / dimensions[0],
        np.array(y_vals) / dimensions[1],
        np.array(z_vals) / dimensions[2],
    )
    grid.dimensions = [
        2 * dimensions_int[1] + 1 - 2 * scale,
        2 * dimensions_int[0] + 1 - 2 * scale,
        2 * dimensions_int[2] + 1 - 2 * scale,
    ]

    # create a structured grid
    # (for this simple example we could've used an unstructured grid too)
    # note the fortran-order call to ravel()!
    # grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["values"] = out_data.flatten()  # also the active scalars

    if scalar != "None":
        scalar_files = glob.glob(scalar + "*bxsf*")
        try:
            scalar_file = scalar_files[0]
            _, _, _, _, _, _, scalar_grid = read_bxsf(
                scalar_file, scale, order, 0, False
            )
            grid.point_data["scalar_field"] = scalar_grid["values"]
        except:
            print("Error: File for scalar fields does not exist")
            exit()

    # calculate gradient
    if fermi_velocity == True:
        grid = grid.compute_derivative(scalars="values")

    iso1 = grid.contour(
        isosurfaces=1,
        rng=[
            e_f + shift_energy,
            e_f + shift_energy,
        ],
        scalars="values",
    )
    if fermi_velocity == True:
        try:
            iso1 = iso1.compute_normals()
            vf = np.einsum(
                "ij,ij->i", iso1.point_data["Normals"], iso1["gradient"]
            )
            if scale > 1:
                iso1.point_data["fermi_velocity"] = -vf
            else:
                iso1.point_data["fermi_velocity"] = vf
        except:
            pass

    isos = [iso1]

    # or: mesh.contour(isosurfaces=np.linspace(10, 40, 3)) etc.

    dimensions = dimensions_int

    # close file
    f.close()

    return k_vectors, eig_vals, e_f, cell, dimensions, isos, grid


def get_brillouin_zone_3d(cell):
    """
    Uses the k-space vectors and voronoi analysis to define
    the BZ of the system

    Args:
        cell: a 3x3 matrix defining the basis vectors in
        reciprocal space

    Returns:
        vor.vertices[bz_vertices]: vertices of BZ
        bz_ridges: edges of the BZ
        bz_facets: BZ facets

    """

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):

        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets


def run_skeaf(file, args):
    """
    Run skeaf on the current bxsf file to determine the orbital
    path for plotting

    Args:
        file: current file to run SKEAF on
        args: cmd line parsed args
    """

    try:
        os.mkdir("./skeaf_out")
    except:
        pass

    # generate config file for ca
    file_io = open(file, "r")

    # get Fermi energy
    for line in file_io:
        if re.search("Fermi Energy:", line):
            line = line.replace(" ", "")
            Ef = line.removeprefix("FermiEnergy:")
            Ef = Ef.strip("\n")
            break

    with open("config.in", "w") as f_out:
        f_out.write(str(file.split("/")[-1]) + "\n")
        f_out.write(str(Ef) + "\n")
        f_out.write(str(args.skeaf_interpolate) + "\n")
        f_out.write(str(args.skeaf_theta) + "\n")
        f_out.write(str(args.skeaf_phi) + "\n")
        f_out.write("n" + "\n")
        f_out.write(str(args.skeaf_min) + "\n")
        f_out.write(str(args.skeaf_min_frac_diff) + "\n")
        f_out.write(str(args.skeaf_min_dist) + "\n")
        f_out.write("y" + "\n")
        f_out.write(str(args.skeaf_theta) + "\n")
        f_out.write(str(args.skeaf_theta) + "\n")
        f_out.write(str(args.skeaf_phi) + "\n")
        f_out.write(str(args.skeaf_phi) + "\n")
        f_out.write("1" + "\n")
    f_out.close()

    popen = subprocess.Popen([r"skeaf", "-rdcfg", "-nodos"])

    popen.wait()


def organise_skeaf(band_index, args):
    """
    Move SKEAF output files into dedicated folder

    Args:
        band-index: current band index being viewed
        args: cmd line parsed args
    """

    os.rename(
        "results_freqvsangle.out",
        f"./skeaf_out/results_freqvsangle.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
    )
    os.rename(
        "results_long.out",
        f"./skeaf_out/results_long.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
    )
    os.rename(
        "results_short.out",
        f"./skeaf_out/results_short.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
    )
    os.rename(
        "results_orbitoutlines_invAng.out",
        f"./skeaf_out/results_orbitoutlines_invAng.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
    )
    os.rename(
        "results_orbitoutlines_invau.out",
        f"./skeaf_out/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
    )


def plot_skeaf():
    """
    Convert SKEAF output to readable form
    for plotting

    Returns:
        orbits_list: list of dataframes containing orbital information
    """

    orbits_file = open("results_orbitoutlines_invau.out").read()
    orbits_split = orbits_file.split("kz")
    orbits_list = []
    for i in range(len(orbits_split) - 1):
        orbit = pd.read_csv(
            io.StringIO(
                orbits_split[i + 1]
                .split("Slice")[0][1:]
                .replace("  ", " ")
                .replace(" ", ",")
            ),
            header=None,
        )
        orbits_list += [orbit]

    return orbits_list


def args_parser():
    """Function to take input command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fermi Surface Plotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        metavar="\b",
        type=str,
        help="Name of the bxsf file in which data is stored",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--interpolation-factor",
        metavar="\b",
        type=int,
        default=1,
        help="Degree of interpolation of Fermi surface",
    )
    parser.add_argument(
        "-b",
        "--bands",
        metavar="\b",
        type=str,
        help="List of bands to include in plot",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        metavar="\b",
        type=float,
        help="Zoom factor of saved image",
        default=1.0,
    )
    parser.add_argument(
        "-se",
        "--shift-energy",
        metavar="\b",
        type=float,
        help="Shift in the Fermi energy from the DFT/input value",
        default=0.0,
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        metavar="\b",
        type=float,
        help="Azimuthal angle of camera",
        default=65,
    )
    parser.add_argument(
        "-e",
        "--elevation",
        metavar="\b",
        type=float,
        help="Elevation angle of camera",
        default=30,
    )
    parser.add_argument(
        "-o",
        "--order",
        metavar="\b",
        type=int,
        help="Order of spline for interpolation between 1 and 5",
        default=3,
    )
    parser.add_argument(
        "-s",
        "--spin",
        metavar="\b",
        type=str,
        choices=["u", "d", "ud", "n"],
        help="Whether the bands are spin polarised and hence which spins to choose. Options: ['u', 'd', 'ud', 'n']",
        default="n",
    )
    parser.add_argument(
        "-vf",
        "--fermi-velocity",
        metavar="\b",
        type=bool,
        help="Compute the Fermi velocity on the surface",
        default=False,
    )
    parser.add_argument(
        "-lw",
        "--line-width",
        metavar="\b",
        type=float,
        help="Linewidth for the Brillouin Zone plot",
        default=2.0,
    )
    parser.add_argument(
        "-op",
        "--opacity",
        metavar="\b",
        type=float,
        help="Opacity of the Fermi surface plot, Range: [0, 1]",
        default=1.0,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        metavar="\b",
        type=float,
        help="Resoltuion of saved pdf image",
        default=1.0,
    )
    parser.add_argument(
        "-int",
        "--interactive",
        metavar="\b",
        type=bool,
        help="Interactive Fermi Surface visualisation",
        default=True,
    )
    parser.add_argument(
        "-bn",
        "--band-name",
        metavar="\b",
        type=bool,
        help="Add Band index to the plots",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        metavar="\b",
        type=str,
        help="The format of the saved figure (png, jpg, pdf). Multiple choice are allowed.",
        choices=["png", "jpg", "pdf"],
        default="pdf",
    )
    parser.add_argument(
        "-pr",
        "--projection",
        metavar="\b",
        type=str,
        help="Whether surfaces are projected in parallel mode or perspective",
        choices=["parallel", "perspective", "par", "per"],
        default="parallel",
    )
    parser.add_argument(
        "-sc",
        "--scalar",
        metavar="\b",
        type=str,
        help="Name of file containing a scalar field to plot on the Fermi surface",
        default="None",
    )
    parser.add_argument(
        "-cmp",
        "--colourmap",
        metavar="\b",
        type=str,
        help="The plot colourmap",
        default="turbo",
    )
    parser.add_argument(
        "-cb",
        "--colourbar",
        metavar="\b",
        type=bool,
        help="Whether to plot colorbar for scalar fields or vf",
        default=False,
    )
    parser.add_argument(
        "-norm",
        "--normalise",
        metavar="\b",
        type=bool,
        help="Whether to renormalise the plot around 0",
        default=False,
    )
    parser.add_argument(
        "-pow",
        "--power",
        metavar="\b",
        type=int,
        help="Power scaling of scalar field",
        default=1,
    )
    parser.add_argument(
        "-nc",
        "--no-clip",
        metavar="\b",
        type=bool,
        help="Whether to clip isosurface",
        default=False,
    )
    parser.add_argument(
        "-sk",
        "--skeaf",
        metavar="\b",
        type=bool,
        help="Whether to run SKEAF to calculate extremal orbital areas",
        default=False,
    )
    parser.add_argument(
        "-sk-t",
        "--skeaf-theta",
        metavar="\b",
        type=float,
        help="Theta angle for extremal loop calculation",
        default=0.0,
    )
    parser.add_argument(
        "-sk-p",
        "--skeaf-phi",
        metavar="\b",
        type=float,
        help="Phi angle for extremal loop calculation",
        default=0.0,
    )
    parser.add_argument(
        "-sk-min",
        "--skeaf-min",
        metavar="\b",
        type=float,
        help="Minimum extremal FS frequency (kT)",
        default=0.0,
    )
    parser.add_argument(
        "-sk-mfd",
        "--skeaf-min-frac-diff",
        metavar="\b",
        type=float,
        help="Maximum fractional diff. between orbit freqs. for averaging",
        default=0.01,
    )
    parser.add_argument(
        "-sk-md",
        "--skeaf-min-dist",
        metavar="\b",
        type=float,
        help="Maximum distance between orbit avg. coords. for averaging",
        default=0.05,
    )
    parser.add_argument(
        "-sk-i",
        "--skeaf-interpolate",
        metavar="\b",
        type=int,
        help="Interpolated number of points per single side",
        default=100,
    )
    parser.add_argument(
        "-sk-lw",
        "--skeaf-linewidth",
        metavar="\b",
        type=float,
        help="Linewidth of SKEAF orbit in plot",
        default=1.5,
    )
    parser.add_argument(
        "-sk-c",
        "--skeaf-colour",
        metavar="\b",
        type=str,
        help="Colour of SKEAF orbit plot",
        default="red",
    )

    return parser


def load_files(args):
    """Function to load in bxsf files taking arguments
    to determine whether we have a spin polarised case

    Args:

        args: command line arguments

    Returns:

        files: list of files in be included in plots
    """

    # load up bxsf files
    if args.bands is not None:
        bands = ast.literal_eval(args.bands)
        files = []
        for band in bands:
            try:
                if args.spin == "u":
                    files += glob.glob(
                        file_name + "*bxsf.band-" + str(band) + "_up"
                    )
                elif args.spin == "d":
                    files += glob.glob(
                        file_name + "*bxsf.band-" + str(band) + "_up"
                    )
                elif args.spin == "ud":
                    files += glob.glob(
                        file_name + "*bxsf.band-" + str(band) + "_*"
                    )
                elif args.spin == "n":
                    files += glob.glob(file_name + "*bxsf.band-" + str(band))
            except:
                print(
                    f"Error: No matching band indices found for band {band}, check band indices"
                )
                exit()
    else:
        files = glob.glob(file_name + "*bxsf.band-*")
        if args.spin == "u":
            files = [file for file in files if file[-2:] == "up"]
        elif args.spin == "d":
            files = [file for file in files if file[-2:] == "dn"]
        elif args.spin == "ud":
            files = [
                file
                for file in files
                if file[-2:] == "dn" or file[-2:] == "up"
            ]
        elif args.spin == "n":
            files = [
                file
                for file in files
                if file[-2:] != "dn" and file[-2:] != "up"
            ]
    return files


# parse command line arguments
parser = args_parser()
args = parser.parse_args()

# assign parsed command line arguments
file_name = args.name
scale = args.interpolation_factor
order = args.order
opacity = args.opacity
formt = args.format

# get the projection
projection = args.projection
if projection == "par":
    projection = "parallel"
if projection == "per":
    projection == "perspective"

# load in bxsf files
files = load_files(args)
# error checking
if len(files) == 0:
    print("Error: No .bxsf files found, check file name")
    exit()

if scale < 1:
    print("Error: Please choose a scaling factor greater than 0.5")
    exit()

if order < 1 or order > 5:
    print("Error: Please choose a interpolation order between 1 aned 5")
    exit()

if opacity > 1 or opacity < 0:
    print("Error: Please choose an opacity between 0 and 1")
    exit()

if formt not in ["png", "jpg", "pdf"]:
    print("Error: File format not allowed")
    exit()

if projection not in ["parallel", "perspective", "par", "per"]:
    print("Error: Projection type not allowed")
    exit()

if args.scalar != "None" and args.fermi_velocity is True:
    print(
        "Error: Cannot plot both the Fermi velocity and a scalar field on the Fermi surface"
    )
    exit()


# initialise 3D visualisation
plotter = pv.Plotter(
    off_screen=True,
    window_size=[
        int(round(args.resolution * 1024)),
        int(round(args.resolution * 768)),
    ],
)

if projection == "parallel":
    plotter.enable_parallel_projection()

if args.interactive == True:
    plotter_int = pv.Plotter()
    if projection == "parallel":
        plotter_int.enable_parallel_projection()

# get cell for FS plot
_, _, _, _, _, _, cell = read_bxsf_info(files[0])


# generate BZ from voronoi analysis
v, e, f = get_brillouin_zone_3d(cell)

# triangulate BZ surface
bz_surf = pv.PolyData(v)
bz_surf = bz_surf.delaunay_3d()
bz_surf = bz_surf.extract_surface()
edges = bz_surf.extract_all_edges()


# make output colorlist
color_list = sns.color_palette("hls", 2 * len(files))
counter = 0

if args.scalar != "None":
    scalar_file = glob.glob(args.scalar + "*bxsf*")[0]
    _, scalar_vals, _, _, _, _, _ = read_bxsf(
        scalar_file,
        1,
        order=1,
        shift_energy=0,
        fermi_velocity=False,
    )

for file in files:

    print(file)

    k_vectors, eig_vals, e_f, cell, dimensions, isos, _ = read_bxsf(
        file,
        scale,
        order=order,
        shift_energy=args.shift_energy,
        fermi_velocity=args.fermi_velocity,
        scalar=args.scalar,
    )

    vec1 = cell[0] * (dimensions[0] - scale) / dimensions[0]
    vec2 = cell[1] * (dimensions[1] - scale) / dimensions[1]
    vec3 = cell[2] * (dimensions[2] - scale) / dimensions[2]

    plotter_ind = pv.Plotter(
        off_screen=True,
        window_size=[
            int(round(args.resolution * 1024)),
            int(round(args.resolution * 768)),
        ],
    )

    if args.scalar != "None" or args.fermi_velocity is True:
        cp = sns.color_palette(args.colourmap, as_cmap=True)

    if projection == "parallel":
        plotter_ind.enable_parallel_projection()

    if args.skeaf == True:
        run_skeaf(file, args)
        orbits_list = plot_skeaf()

    for iso in isos:

        try:

            if args.no_clip == False:

                iso = iso.clip_surface(bz_surf, invert=True)

            if args.fermi_velocity == True:

                plotter_ind.add_mesh(
                    iso,
                    lighting=True,
                    scalars="fermi_velocity",
                    cmap=cp,
                    opacity=1.0,
                )
                plotter.add_mesh(
                    iso,
                    lighting=True,
                    scalars="fermi_velocity",
                    cmap=cp,
                    opacity=1.0,
                )
                if args.colourbar == False:
                    plotter.remove_scalar_bar()
                    plotter_ind.remove_scalar_bar()

                if args.interactive == True:
                    plotter_int.add_mesh(
                        iso,
                        lighting=True,
                        scalars="fermi_velocity",
                        cmap=cp,
                        opacity=1.0,
                    )
                    if args.colourbar == False:
                        plotter_int.remove_scalar_bar()

            elif args.scalar != "None":
                if args.power > 1:
                    mask = np.where(iso["scalar_field"] >= 0, 1, -1)
                    iso["scalar_field"] = iso["scalar_field"] ** args.power
                    if args.power % 2 == 0:
                        iso["scalar_field"] = iso["scalar_field"] * mask

                if args.normalise == True:
                    iso["scalar_field"] = iso["scalar_field"] / max(
                        abs(scalar_vals.flatten().min()),
                        abs(scalar_vals.flatten().min()),
                    )
                    c_max = iso["scalar_field"].min()
                    c_min = iso["scalar_field"].max()

                    abs_max = max(abs(c_max), abs(c_min))
                    ratio_max = c_max / (2 * abs_max)
                    ratio_min = c_min / (2 * abs_max)

                    cp = shiftedColorMap(
                        cp,
                        start=0.5 + ratio_min,
                        midpoint=(1.0 + ratio_max + ratio_min) / 2,
                        stop=0.5 + ratio_max,
                        name=f"{file}",
                    )
                    cp = cp.reversed()

                plotter_ind.add_mesh(
                    iso,
                    lighting=True,
                    scalars="scalar_field",
                    cmap=cp,
                    opacity=1.0,
                )
                plotter.add_mesh(
                    iso,
                    lighting=True,
                    scalars="scalar_field",
                    cmap=cp,
                    opacity=1.0,
                )

                if args.colourbar == False:
                    plotter.remove_scalar_bar()
                    plotter_ind.remove_scalar_bar()

                if args.interactive == True:
                    plotter_int.add_mesh(
                        iso,
                        lighting=True,
                        scalars="scalar_field",
                        cmap=cp,
                        opacity=1.0,
                    )
                    if args.colourbar == False:
                        plotter_int.remove_scalar_bar()

            else:
                plotter.add_mesh(
                    iso,
                    lighting=True,
                    color=color_list[2 * counter],
                    opacity=args.opacity,
                    backface_params={"color": color_list[2 * counter + 1]},
                )
                plotter_ind.add_mesh(
                    iso,
                    lighting=True,
                    color=color_list[2 * counter],
                    opacity=args.opacity,
                    backface_params={"color": color_list[2 * counter + 1]},
                )
                if args.interactive == True:
                    plotter_int.add_mesh(
                        iso,
                        lighting=True,
                        color=color_list[2 * counter],
                        opacity=args.opacity,
                        backface_params={"color": color_list[2 * counter + 1]},
                    )

        except:
            # print(file + " contains an empty mesh")
            pass

        for xx in e:
            line = pv.MultipleLines(
                points=np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T
            )
            plotter_ind.add_mesh(
                line,
                color="black",
                line_width=args.resolution * args.line_width,
            )

        if args.skeaf == True:
            for orbit in orbits_list:
                orbit_line = pv.MultipleLines(
                    points=np.array(
                        [
                            orbit[1] / (2 * np.pi),
                            orbit[2] / (2 * np.pi),
                            orbit[3] / (2 * np.pi),
                        ]
                    ).T
                )
                plotter.add_mesh(
                    orbit_line,
                    color=args.skeaf_colour,
                    line_width=args.resolution
                    * args.line_width
                    * args.skeaf_linewidth,
                )
                plotter_ind.add_mesh(
                    orbit_line,
                    color=args.skeaf_colour,
                    line_width=args.resolution
                    * args.line_width
                    * args.skeaf_linewidth,
                )
                if args.interactive == True:
                    plotter_int.add_mesh(
                        orbit_line,
                        color=args.skeaf_colour,
                        line_width=args.resolution
                        * args.line_width
                        * args.skeaf_linewidth,
                    )

        plotter_ind.set_background("white")
        plotter_ind.camera_position = "yz"
        plotter_ind.set_position([0.5 / args.zoom, 0, 0])
        plotter_ind.camera.azimuth = args.azimuth
        plotter_ind.camera.elevation = args.elevation
        band_index = file.split(".")[-1]
        if args.band_name == True:
            plotter_ind.add_title("Band " + band_index.split("-")[1])

        if args.skeaf == True:

            organise_skeaf(band_index, args)

        # Save individual plots
        if formt == "pdf":
            plotter_ind.save_graphic("FS_side_" + band_index + ".pdf")
        elif formt == "png":
            plotter_ind.screenshot("FS_side_" + band_index + ".png")
        elif formt == "jpg":
            plotter_ind.screenshot("FS_side_" + band_index + ".jpg")

        counter += 1


# plot BZ
for xx in e:
    line = pv.MultipleLines(points=np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T)
    plotter.add_mesh(
        line, color="black", line_width=args.resolution * args.line_width
    )
    if args.interactive == True:
        plotter_int.add_mesh(line, color="black", line_width=args.line_width)


plotter.set_background("white")
plotter.camera_position = "yz"
plotter.set_position([0.5 / args.zoom, 0, 0])
plotter.camera.azimuth = args.azimuth
plotter.camera.elevation = args.elevation
if args.band_name == True:
    plotter.add_title("Total Fermi Surface")

# Save overall plot
if formt == "pdf":
    plotter.save_graphic("FS.pdf")
elif formt == "png":
    plotter.screenshot("FS.png")
elif formt == "jpg":
    plotter.screenshot("FS.jpg")

if args.interactive == True:
    plotter_int.set_background("white")
    plotter_int.camera_position = "yz"
    plotter_int.set_position([0.5 / args.zoom, 0, 0])
    plotter_int.camera.azimuth = args.azimuth
    plotter_int.camera.elevation = args.elevation
    plotter_int.show()

    camera_coord = np.array(
        [
            plotter_int.camera.position[0],
            plotter_int.camera.position[1],
            plotter_int.camera.position[2],
        ]
    )

    if plotter_int.camera.position[1] != 0:
        elevation_rad = np.arctan(
            plotter_int.camera.position[2] / plotter_int.camera.position[1]
        )
        elevation_deg = np.degrees(elevation_rad)
    else:
        elevation_deg = 90

    if plotter_int.camera.position[1] != 0:
        azimuth_rad = np.arctan(
            plotter_int.camera.position[0] / plotter_int.camera.position[1]
        )
        azimuth_deg = np.degrees(azimuth_rad)
    else:
        azimuth_deg = 90

    zoom = 0.5 / np.linalg.norm(camera_coord)

    print("\nFinal Camera coordinates were:")
    print(f"\tAzimuthal angle: {azimuth_deg}")
    print(f"\tElevation: {elevation_deg}")
    print(f"\tZoom: {zoom}")
