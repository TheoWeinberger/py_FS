import os
import re
import subprocess
import pandas as pd
from scipy.interpolate import splprep, splev
import numpy as np

def run_skeaf(file, band_index, args):
    """
    Run skeaf on the current bxsf file to determine the orbital
    path for plotting

    Args:
        file: current file to run SKEAF on
        band_index: current band index being viewed
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
        f_out.write(str(float(Ef) + args.shift_energy) + "\n")
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

    if args.shift_energy == 0.0:

        try:
            os.rename(
                f"./skeaf_out/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
                "results_orbitoutlines_invau.out",
            )
        except:

            popen = subprocess.Popen([r"skeaf", "-rdcfg", "-nodos"])

            popen.wait()

    else:

        try:
            os.rename(
                f"./skeaf_out/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
                "results_orbitoutlines_invau.out",
            )
        except:

            popen = subprocess.Popen([r"skeaf", "-rdcfg", "-nodos"])

            popen.wait()


def organise_skeaf(band_index, args):
    """
    Move SKEAF output files into dedicated folder

    Args:
        band-index: current band index being viewed
        args: cmd line parsed args
    """

    if args.shift_energy == 0.0:

        try:
            os.rename(
                "results_freqvsangle.out",
                f"./skeaf_out/results_freqvsangle.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_long.out",
                f"./skeaf_out/results_long.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_short.out",
                f"./skeaf_out/results_short.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invAng.out",
                f"./skeaf_out/results_orbitoutlines_invAng.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invau.out",
                f"./skeaf_out/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass

    else:
        try:
            os.rename(
                "results_freqvsangle.out",
                f"./skeaf_out/results_freqvsangle.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_long.out",
                f"./skeaf_out/results_long.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_short.out",
                f"./skeaf_out/results_short.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invAng.out",
                f"./skeaf_out/results_orbitoutlines_invAng.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invau.out",
                f"./skeaf_out/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass


def plot_skeaf(args):
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
        orbit.loc[len(orbit.index)] = orbit.loc[0]
        if args.skeaf_line_interpolate != 1:
            tck, u = splprep(
                [orbit[1], orbit[2], orbit[3]],
                k=args.skeaf_line_order,
                s=args.skeaf_line_smoothness,
            )
            u = np.linspace(
                0, 1, len(u) * args.skeaf_line_interpolate, endpoint=True
            )
            new_points = splev(u, tck)
            orbit_int = pd.DataFrame([])
            orbit_int[0] = np.zeros_like(new_points[0])
            orbit_int[1] = new_points[0]
            orbit_int[2] = new_points[1]
            orbit_int[3] = new_points[2]
            orbits_list += [orbit_int]
        else:
            orbits_list += [orbit]

    return orbits_list
