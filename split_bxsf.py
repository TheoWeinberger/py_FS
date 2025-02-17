#!/usr/bin/env python3
"""
Copyright 2025 T I Weinberger

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Splits bxsf files consisting of all bands to 
produce bxsf files plottable by fs_plot.py
"""


import numpy as np
import re
import glob
import os
import subprocess
import io
import argparse
import ast
import pandas as pd


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
        "-b",
        "--bands",
        metavar="\b",
        type=str,
        help="List of bands to split from bxsf file. If None, only bands crossing Ef will be split",
    )
    return parser


def read_bxsf_info(file_name):

    # open and read file
    f = open(file_name, "r")
    lines = f.readlines()

    """
    this corresponds to code for getting the fermi energy
    """
    # get fermi energy
    for line in lines:
        if "Fermi Energy" in line:
            x = line.split(" ")
            x = [val for _, val in enumerate(x) if val != ""]
            x = [val for _, val in enumerate(x) if val != "\t"]
            x = x[-1].split("\n")
            # convert ef to a number
            ef = float(x[0])
            break

    """
    Now get the dimensions of the k mesh
    """
    line_counter = 0
    for line in lines:
        if "BAND:" in line:
            x = re.findall(r"\d+", lines[line_counter - 5])
            # convert dimensions to integer values
            dimensions = [int(dim) for dim in x]
            """
            Get basis vectors
            """
            # first spanning vector
            vec1 = lines[line_counter - 1].split(" ")
            vec1 = [val.strip("\n") for val in vec1]
            vec1 = [val.strip("\t") for val in vec1]
            vec1 = [val for _, val in enumerate(vec1) if val != ""]
            # convert dimensions to float values
            vec1 = [float(val) for val in vec1]

            # second spanning vector
            vec2 = lines[line_counter - 2].split(" ")
            vec2 = [val.strip("\n") for val in vec2]
            vec2 = [val.strip("\t") for val in vec2]
            vec2 = [val for _, val in enumerate(vec2) if val != ""]
            # convert dimensions to float values
            vec2 = [float(val) for val in vec2]

            # thrid spanning vector
            vec3 = lines[line_counter - 3].split(" ")
            vec3 = [val.strip("\n") for val in vec3]
            vec3 = [val.strip("\t") for val in vec3]
            vec3 = [val for _, val in enumerate(vec3) if val != ""]
            # convert dimensions to float values
            vec3 = [float(val) for val in vec3]

            # shift
            shift = lines[line_counter - 4].split(" ")
            shift = [val.strip("\n") for val in shift]
            shift = [val.strip("\t") for val in shift]
            shift = [val for _, val in enumerate(shift) if val != ""]
            # convert dimensions to float values
            shift = [float(val) for val in shift]

            # n_bands
            n_bands = lines[line_counter - 6].split(" ")
            n_bands = [val.strip("\n") for val in n_bands]
            n_bands = [val.strip("\t") for val in n_bands]
            n_bands = [val for _, val in enumerate(n_bands) if val != ""]
            # convert dimensions to float values
            n_bands = [int(val) for val in n_bands]
            n_bands = n_bands[0]
            break

        line_counter += 1

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    vec3 = np.array(vec3)

    cell = np.array([vec1, vec2, vec3])

    # close file
    f.close()

    return vec1, vec2, vec3, dimensions, ef, shift, n_bands


def load_file(args):
    """Function to load in bxsf files taking arguments
    to determine whether we have a spin polarised case

    Args:

        args: command line arguments

    Returns:

        file: bxsf file to be split
    """

    # load up bxsf files
    try:
        file = glob.glob(args.name + "*.bxsf")
    except:
        print(f"Error: No matching bxsf found with name {args.name}")
        exit()

    if len(file) == 0:
        print(f"Error: No matching bxsf found with name {args.name}")
        exit()
    return file


def get_energies_band(band, file):
    # open and read file
    f = open(file, "r")
    lines = f.readlines()

    # get everything after band index
    lines_conc = "".join(lines)
    found = False
    for line in lines:
        if "BAND:" in line:
            if f" {band}\n" in line or f"\t{band}\n" in line:
                eigen_text = lines_conc.split(line)[1]
                found = True
                break
    if found == True:
        eigen_text = eigen_text.split("END_BANDGRID_3D")[0]
        eigen_text = eigen_text.split("BAND:")[0]
        eigen_text = re.sub(" +", " ", eigen_text)
        eigen_vals = eigen_text.split(" ")
        eigen_vals = [val.strip("\n") for val in eigen_vals]
        eigen_vals = eigen_vals[1:-1]
        eigen_vals = [float(val) for val in eigen_vals]
    else:
        print(f"Error: No matching band index found with index {band}")
        exit()

    f.close()

    return eigen_vals


def gen_bxsf(args, eigen_vals, ef, band, shift, dimensions, vec1, vec2, vec3):

    f = open(args.name + ".bxsf.band-" + str(band), "w")

    """
        general .bxsf rubbish
        """
    f.write(" BEGIN_INFO\n")
    f.write("  Fermi Energy:     " + str(ef) + "\n")
    f.write(" END_INFO\n")
    f.write(" BEGIN_BLOCK_BANDGRID_3D\n")
    f.write(" band_energies\n")
    f.write(" BANDGRID_3D_BANDS\n")
    f.write(" 1\n")
    f.write(
        " "
        + str(dimensions[0])
        + " "
        + str(dimensions[1])
        + " "
        + str(dimensions[2])
        + "\n"
    )

    f.write(f"\t {shift[0]:.6f}\t {shift[1]:.6f}\t {shift[2]:.6f}\n")

    vec_list = [vec1, vec2, vec3]

    for vec in vec_list:

        f.write(f"\t {vec[0]:.6f}\t {vec[1]:.6f}\t {vec[2]:.6f}\n")

    f.write(" BAND:   " + str(band) + "\n")

    counter = 0
    for eig in eigen_vals:
        f.write(f" {eig:.6f}")
        if (counter + 1) % 6 == 0:
            f.write("\n")
        counter += 1

    # last new line
    if (counter + 1) % 6 == 0:
        pass
    else:
        f.write("\n")

    f.write(" END_BANDGRID_3D\n")
    f.write(" END_BLOCK_BANDGRID_3D\n")
    f.close()


def write_bxsf(args, file, vec1, vec2, vec3, dimensions, ef, shift, n_bands):

    if args.bands is not None:
        for band in ast.literal_eval(args.bands):
            eig_vals = get_energies_band(str(band), file)
            gen_bxsf(
                args, eig_vals, ef, band, shift, dimensions, vec1, vec2, vec3
            )

    else:
        closest_val = 10000
        closest_ind = 1
        metal = False
        for i in range(n_bands):
            eig_vals = get_energies_band(str(i + 1), file)
            if min(eig_vals) < ef and max(eig_vals) > ef:
                gen_bxsf(
                    args,
                    eig_vals,
                    ef,
                    i + 1,
                    shift,
                    dimensions,
                    vec1,
                    vec2,
                    vec3,
                )
                metal = True
            else:
                closest_val_test = min(
                    abs(min(eig_vals) - ef), abs(max(eig_vals) - ef)
                )
                if min(eig_vals) < ef:
                    sign = -1
                else:
                    sign = +1
                if closest_val_test < abs(closest_val):
                    closest_val = sign * closest_val_test
                    closest_ind = i + 1
        if metal == False:
            print(
                f"No matching band crossing the Fermi level found\n"
                f"The closest band is band index {closest_ind}, which is {closest_val} away from Ef in the units of your file"
            )
            exit()


def main():
    parser = args_parser()
    args = parser.parse_args()

    file = load_file(args)
    vec1, vec2, vec3, dimensions, ef, shift, n_bands = read_bxsf_info(file[0])

    write_bxsf(args, file[0], vec1, vec2, vec3, dimensions, ef, shift, n_bands)


main()
