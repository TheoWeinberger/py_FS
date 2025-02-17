import os
import re
import subprocess
import pandas as pd
from scipy.interpolate import splprep, splev
import numpy as np
import io


def check_voronoi(P, a1, a2, a3, lattice_points_range=2):
    """
    Checks if the point P lies within the Voronoi tile centered at the origin,
    spanned by the lattice vectors a1, a2, and a3 in 3D space.

    Parameters:
    P (np.array): The point to check (3D vector).
    a1 (np.array): The first lattice vector (3D vector).
    a2 (np.array): The second lattice vector (3D vector).
    a3 (np.array): The third lattice vector (3D vector).
    lattice_points_range (int): The range of (m, n, p) values to check lattice points.

    Returns:
    bool: True if the point lies inside the Voronoi tile, False otherwise.
    """

    # Check distances to all lattice points (m * a1 + n * a2 + p * a3)
    indices = []
    for m in range(-lattice_points_range, lattice_points_range + 1):
        for n in range(-lattice_points_range, lattice_points_range + 1):
            for p in range(-lattice_points_range, lattice_points_range + 1):
                if m == 0 and n == 0 and p == 0:
                    continue  # Skip the origin itself
                if np.linalg.norm(
                    P - (m * a1 + n * a2 + p * a3)
                ) < np.linalg.norm(P):
                    indices.append([m, n, p])
    sorted_indices = sorted(
        indices,
        key=lambda x: np.linalg.norm(P - (x[0] * a1 + x[1] * a2 + x[2] * a3)),
    )

    if len(indices) > 0:
        return np.array(sorted_indices[0])
    elif len(indices) == 0:
        return np.array([0, 0, 0])


def modify_invau(file_path, dataframes):
    """
    Replaces numerical values below the kx, ky, kz row in a file with corresponding values from the provided dataframes.

    :param file_path: str, path to the file to read and modify in place.
    :param dataframes: list of pd.DataFrame, new data to replace the kx, ky, kz rows in each slice.
    """

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize variables
    new_lines = []
    dataframe_index = 0
    inside_k_points = False

    for line in lines:
        if "Slice" in line:
            inside_k_points = False
            new_lines.append(line)
        elif "Points" in line:
            inside_k_points = False
            new_lines.append(line)
        elif "kx" in line:
            # When encountering the kx, ky, kz header, enable k-points replacement
            inside_k_points = True
            new_lines.append(line)

            # Add data from the corresponding DataFrame
            if dataframe_index < len(dataframes):
                df = dataframes[dataframe_index]
                for _, row in df.iterrows():
                    new_lines.append(
                        f" {row[1]:.6E} {row[2]:.6E} {row[3]:.6E}\n"
                    )
                dataframe_index += 1
            else:
                raise ValueError(
                    "Not enough DataFrames provided for the number of slices."
                )
        elif inside_k_points and re.match(
            r"\s*-?\d+\.\d+E[+-]\d+\s+-?\d+\.\d+E[+-]\d+\s+-?\d+\.\d+E[+-]\d+",
            line,
        ):
            continue
        else:
            new_lines.append(line)

    # Write the modified content back to the same file
    with open(file_path, "w") as file:
        file.writelines(new_lines)


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
                f"./skeaf_out/other/results_freqvsangle.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_long.out",
                f"./skeaf_out/other/results_long.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_short.out",
                f"./skeaf_out/other/results_short.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invAng.out",
                f"./skeaf_out/other/results_orbitoutlines_invAng.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out",
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
                f"./skeaf_out/other/results_freqvsangle.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_long.out",
                f"./skeaf_out/other/results_long.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_short.out",
                f"./skeaf_out/other/results_short.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invAng.out",
                f"./skeaf_out/other/results_orbitoutlines_invAng.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass
        try:
            os.rename(
                "results_orbitoutlines_invau.out",
                f"./skeaf_out/other/results_orbitoutlines_invau.{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}_se_{args.shift_energy}.out",
            )
        except:
            pass


def plot_skeaf(band_index, args, cell):
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

    ##############################################################################
    # Make sure all orbits lie in the first BZ

    mod_orbits_list = []
    for orbit in orbits_list:
        o = orbit.to_numpy()[:, 1:] / (2 * np.pi)

        # Change basis from conventional to reciprocal coordinates
        o_basis_ch = np.array(
            [np.linalg.inv(cell.T).dot(i).tolist() for i in o]
        )

        # Determine one point on a given orbit that will be used to find the shift in k-space
        lowest_vector = min(o, key=lambda v: np.linalg.norm(v))
        cv = check_voronoi(lowest_vector, cell[0, :], cell[1, :], cell[2, :])

        # Shift in k-space so that point lies within the first BZ
        o_basis_ch_inside = o_basis_ch
        for i in range(len(o[:, 0])):
            if np.any(cv != 0):
                o_basis_ch_inside[i, 0] = o_basis_ch[i, 0] - cv[0]
                o_basis_ch_inside[i, 1] = o_basis_ch[i, 1] - cv[1]
                o_basis_ch_inside[i, 2] = o_basis_ch[i, 2] - cv[2]

        # Change basis back to conventional coordinates
        o_inside = np.array(
            [(cell.T.dot(i.T)).tolist() for i in o_basis_ch_inside]
        )

        orbit = orbit.to_numpy()
        orbit[:, 1:] = o_inside * (2 * np.pi)
        orbit = pd.DataFrame(orbit)
        mod_orbits_list += [orbit.iloc[:, 1:]]

    ##############################################################################
    modify_invau("results_orbitoutlines_invau.out", mod_orbits_list)

    return orbits_list
