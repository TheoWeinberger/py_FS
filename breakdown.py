import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.constants as ct
from scipy.spatial import Voronoi
from src.utils import (
    get_brillouin_zone_3d,
    read_bxsf_info,
    load_files,
    read_bxsf,
)
import argparse
import ast
import glob
import seaborn as sns


def args_parser_breakdown() -> argparse.ArgumentParser:
    """
    Function to take input command line arguments

    Returns:
        argparse.ArgumentParser: Argument parser for the breakdown evaluator
    """
    parser = argparse.ArgumentParser(
        description="Breakdown Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        metavar="\b",
        type=str,
        help="Name of the bxsf file from which the quantum oscillations were calculated",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--bands",
        metavar="\b",
        type=str,
        required=True,
        help="List of bands with skeaf derived orbitals to compare for breakdown (note these can be the same file). Can be at most two.",
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
        "-i",
        "--interpolation",
        metavar="\b",
        type=int,
        default=3,
        help="Degree of interpolation for the orbits",
    )
    parser.add_argument(
        "-u",
        "--units",
        metavar="\b",
        type=str,
        default="au",
        choices=["au", "ang"],
        help="Units of the orbits (either atomic units or Angstroms)",
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
        "-p",
        "--plot",
        metavar="\b",
        type=bool,
        help="Plot Fermi surfaces and loops",
        default=False,
    )
    parser.add_argument(
        "-pi",
        "--plot-interpolate",
        metavar="\b",
        type=int,
        help="Interpolation factor for Fermi surface plot",
        default=1,
    )
    parser.add_argument(
        "-po",
        "--plot-order",
        metavar="\b",
        type=int,
        help="Order of interpolation for Fermi surface plot",
        default=1,
    )
    parser.add_argument(
        "-op",
        "--opacity",
        metavar="\b",
        type=float,
        help="Opacity of Fermi surface plot",
        default=0.4,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        metavar="\b",
        type=float,
        help="Resolution of Fermi surface plot",
        default=1.0,
    )
    parser.add_argument(
        "-l1c",
        "--loop1-colour",
        metavar="\b",
        type=str,
        help="Colour of first loop",
        default="red",
    )
    parser.add_argument(
        "-l2c",
        "--loop2-colour",
        metavar="\b",
        type=str,
        help="Colour of second loop",
        default="red",
    )
    parser.add_argument(
        "-ll",
        "--loop-linewidth",
        metavar="\b",
        type=str,
        help="Width of loop lines",
        default=6.0,
    )
    parser.add_argument(
        "-cc",
        "--connect-colour",
        metavar="\b",
        type=str,
        help="Colour of first loop",
        default="black",
    )
    parser.add_argument(
        "-cl",
        "--connect-linewidth",
        metavar="\b",
        type=str,
        help="Width of connect lines",
        default=3.0,
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
        "-z",
        "--zoom",
        metavar="\b",
        type=float,
        help="Zoom factor of saved image",
        default=1.0,
    )
    return parser


def get_skeaf_file(args: argparse.Namespace, band_index: int) -> str:
    """
    Get the skeaf file for a given band index

    Args:
        args (argparse.Namespace): Parsed command line arguments
        band_index (int): Index of the band

    Returns:
        str: Path to the skeaf file
    """
    file = glob.glob(
        f"./skeaf_out/results_orbitoutlines_invau.band-{band_index}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}.out"
    )

    return file[0]


# Function to read the loop data from the file
def read_loops(
    file_path: str, interpol: int, units: str = "au"
) -> tuple[list[np.ndarray], list[str]]:
    """
    Read the loop data from the file

    Args:
        file_path (str): Path to the file containing loop data
        interpol (int): Degree of interpolation for the orbits
        units (str, optional): Units of the orbits (either atomic units or Angstroms). Defaults to "au".

    Returns:
        tuple[list[np.ndarray], list[str]]: List of loops and corresponding slice names
    """
    if units == "au":
        conversion_factor = (
            0.52917721067  # Conversion factor for units (Bohr to Angstroms)
        )
    else:
        conversion_factor = 1.0
    loops = []
    slices = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        loop = []
        for line in lines:
            if line.strip().startswith(
                "Slice"
            ):  # take slice name before starting loop coords
                slice_name = line.strip().split("=")[1].split(",")[0].strip(" ").strip("\t")
                slices += [slice_name]
            if line.strip().startswith("Points"):  # Start of a new loop
                if loop:
                    ###################################
                    l = np.array(loop)
                    t = np.arange(len(loop))
                    spline_x = CubicSpline(t, l[:, 0], bc_type="periodic")
                    spline_y = CubicSpline(t, l[:, 1], bc_type="periodic")
                    spline_z = CubicSpline(t, l[:, 2], bc_type="periodic")
                    t_fine = np.linspace(0, len(loop), interpol * len(loop))
                    loop = [
                        [
                            float(spline_x(t_fine)[i]),
                            float(spline_y(t_fine)[i]),
                            float(spline_z(t_fine)[i]),
                        ]
                        for i in range(len(t_fine))
                    ]
                    loop.append(loop[0])
                    ###################################
                    loops.append(np.array(loop))
                    loop = []
            elif len(line.strip().split()) == 3:  # kx, ky, kz values
                try:
                    coords = list(map(float, line.split()))
                    coords = [
                        1e10 * c / conversion_factor for c in coords
                    ]  # Apply unit conversion to 1/m
                    loop.append(coords)
                except ValueError:
                    pass

        if loop:  # Add the last loop if it exists
            ###################################
            slices += [slice_name]
            l = np.array(loop)
            t = np.arange(len(loop))
            spline_x = CubicSpline(t, l[:, 0], bc_type="periodic")
            spline_y = CubicSpline(t, l[:, 1], bc_type="periodic")
            spline_z = CubicSpline(t, l[:, 2], bc_type="periodic")
            t_fine = np.linspace(0, len(loop), interpol * len(loop))
            loop = [
                [
                    float(spline_x(t_fine)[i]),
                    float(spline_y(t_fine)[i]),
                    float(spline_z(t_fine)[i]),
                ]
                for i in range(len(t_fine))
            ]
            loop.append(loop[0])
            ###################################
            loops.append(np.array(loop))
    return loops, slices


# Function to check if two loops are in the same plane
def are_loops_in_same_plane(
    loop1: np.ndarray, loop2: np.ndarray
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Check if two loops are in the same plane

    Args:
        loop1 (np.ndarray): First loop
        loop2 (np.ndarray): Second loop

    Returns:
        tuple[bool, np.ndarray, np.ndarray]: Boolean indicating if loops are in the same plane, normal vector, and centroid of the plane
    """

    def fit_plane(loop):
        # Fit a plane using Singular Value Decomposition (SVD)
        centroid = np.mean(loop, axis=0)
        _, _, vh = np.linalg.svd(loop - centroid)
        normal_vector = vh[-1]  # Plane normal is the last row of V^H
        return centroid, normal_vector

    centroid1, normal1 = fit_plane(loop1)
    centroid2, normal2 = fit_plane(loop2)

    # Check if normal vectors are parallel (or anti-parallel)
    parallel_check = np.isclose(
        np.abs(np.dot(normal1, normal2)), 1.0, atol=1e-3
    )

    # Check if loop2 lies in the plane of loop1
    distance_check = np.allclose(
        np.dot(loop2 - centroid1, normal1), 0, atol=1e8
    )

    return parallel_check and distance_check, normal1, centroid1


# Function to find the closest points between two loops
def find_closest_points(
    loop1: np.ndarray, loop2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the closest points between two loops

    Args:
        loop1 (np.ndarray): First loop
        loop2 (np.ndarray): Second loop

    Returns:
        tuple[np.ndarray, np.ndarray]: Closest points on the two loops
    """
    min_dist = float("inf")
    closest_points = (None, None)
    for p1 in loop1:
        for p2 in loop2:
            dist = np.linalg.norm(p1 - p2)
            if dist < min_dist:
                min_dist = dist
                closest_points = (p1, p2)
    return closest_points


# Function to calculate radius of curvature using spline interpolation
def calculate_radius_of_curvature_spline(
    loop: np.ndarray, point: np.ndarray
) -> float:
    """
    Calculate radius of curvature using spline interpolation

    Args:
        loop (np.ndarray): Loop data
        point (np.ndarray): Point on the loop

    Returns:
        float: Radius of curvature at the given point
    """
    # Find the closest point index to the given point
    point_idx = np.argmin(np.linalg.norm(loop - point, axis=1))

    # Parameterize the loop
    t = np.arange(len(loop))
    spline_x = CubicSpline(t, loop[:, 0], bc_type="periodic")
    spline_y = CubicSpline(t, loop[:, 1], bc_type="periodic")
    spline_z = CubicSpline(t, loop[:, 2], bc_type="periodic")

    # Calculate derivatives at the point
    t_point = t[point_idx]
    r_prime = np.array(
        [spline_x(t_point, 1), spline_y(t_point, 1), spline_z(t_point, 1)]
    )
    r_double_prime = np.array(
        [spline_x(t_point, 2), spline_y(t_point, 2), spline_z(t_point, 2)]
    )

    # Compute radius of curvature
    norm_r_prime = np.linalg.norm(r_prime)
    norm_cross = np.linalg.norm(np.cross(r_prime, r_double_prime))
    radius_of_curvature = (
        (norm_r_prime**3) / norm_cross if norm_cross != 0 else float("inf")
    )

    return radius_of_curvature


# Function to plot the loops and closest points
def analyse_loops(
    loop1: np.ndarray,
    loop2: np.ndarray,
    points: tuple[np.ndarray, np.ndarray],
    dist: float,
    normal: np.ndarray,
    centroid: np.ndarray,
) -> None:
    """
    Plot the loops and closest points

    Args:
        loop1 (np.ndarray): First loop
        loop2 (np.ndarray): Second loop
        points (tuple[np.ndarray, np.ndarray]): Closest points on the two loops
        dist (float): Distance between the closest points
        normal (np.ndarray): Normal vector of the plane
        centroid (np.ndarray): Centroid of the plane
    """

    # # Highlight the closest points
    p1, p2 = points
 
    # Calculate and annotate radius of curvature
    radius1 = calculate_radius_of_curvature_spline(loop1, p1)
    radius2 = calculate_radius_of_curvature_spline(loop2, p2)

    # Handle infinite radius gracefully for B0 calculation
    if (radius1 != float("inf")) and (radius2 != float("inf")):
        critical_field = (
            (np.pi / 2)
            * (ct.hbar / ct.e)
            * ((dist) ** 3 / (1 / radius1 + 1 / radius2)) ** 0.5
        )
        critical_field_label = f"$B_0$ = {critical_field:.1f} T"
    else:
        critical_field = None
        critical_field_label = f"$B_0$ = Undefined (Infinite Radius)"

    critical_field_approx = (np.pi / 2) * (ct.hbar / ct.e) * (dist**4) ** 0.5
    critical_field_approx_label = f"$B_0^a$ = {critical_field_approx:.1f} T"

    print(f"Distance: {dist:.4f} 1/m")
    print(f"Radius of curvature 1: {radius1:.4f} 1/m")
    print(f"Radius of curvature 2: {radius2:.4f} 1/m")
    if critical_field is not None:
        print(critical_field_label + "\n" + critical_field_approx_label)
    else:
        print("Critical Field: Undefined (Infinite Radius)")

# Main function to execute the program
def main() -> None:
    """
    Main function to execute the program
    """
    # Load in arguments
    args = args_parser_breakdown().parse_args()
    # Load bxsf file details
    # load in bxsf files
    bxsf_files = load_files(args)
    # error checking
    if len(bxsf_files) == 0:
        print("Error: No .bxsf files found, check file name")
        exit()

    # get cell for FS plot
    _, _, _, _, _, _, cell = read_bxsf_info(bxsf_files[0])

    # generate BZ from voronoi analysis
    v, e, f = get_brillouin_zone_3d(cell)
    # triangulate BZ surface
    bz_surf = pv.PolyData(v)
    bz_surf = bz_surf.delaunay_3d()
    bz_surf = bz_surf.extract_surface()
    edges = bz_surf.extract_all_edges()

    band_list = ast.literal_eval(args.bands)
    if len(band_list) > 2:
        print("Error: Can only compare at most two bands")
        exit()
    else:
        if len(band_list) == 1:
            band_list.append(band_list[0])

    orbit_files = []
    for band in band_list:
        orbit_files.append(get_skeaf_file(args, band))

    loops_comp = []
    slice_comp = []
    # Read loops from file
    for loop_file_name in orbit_files:
        loop_t, slice_t = read_loops(
            loop_file_name, args.interpolation, args.units
        )
        loops_comp += [loop_t]
        slice_comp += [slice_t]

    counter1 = 0
    for loop1 in loops_comp[0]:
        counter2 = 0
        for loop2 in loops_comp[1]:
            print(
                f"Analysing orbits from band {band_list[0]}, loop slice {slice_comp[0][counter1]} and {band_list[1]}, loop slice {slice_comp[1][counter2]}\n"
            )
            # Check if loops are in the same plane
            plane_check, normal, centroid = are_loops_in_same_plane(
                loop1, loop2
            )
            if not plane_check:
                print(
                    f"Loops for band {band_list[0]}, loop slice {slice_comp[0][counter1]} are not in the same plane as band {band_list[1]}, loop slice {slice_comp[1][counter2]}\n"
                )
                print("\n")
                print("\n")
            else:
                # Find the closest points between the two loops
                points = find_closest_points(loop1, loop2)
                dist = np.linalg.norm(points[0] - points[1])

                # Plot the loops and closest points
                analyse_loops(loop1, loop2, points, dist, normal, centroid)
                print("\n")
                print("\n")

                if args.plot == True:
                    # initialise 3D visualisation
                    plotter = pv.Plotter(
                        window_size=[
                            int(round(args.resolution * 1024)),
                            int(round(args.resolution * 768)),
                        ],
                    )

                    # make output colorlist
                    color_list = sns.color_palette("hls", 2 * len(bxsf_files))
                    counter = 0

                    for file in bxsf_files:

                        k_vectors, eig_vals, e_f, cell, dimensions, isos, _ = (
                            read_bxsf(
                                file,
                                args.plot_interpolate,
                                order=args.plot_order,
                                shift_energy=args.shift_energy,
                                fermi_velocity=False,
                            )
                        )

                        vec1 = cell[0] * (dimensions[0] - 1) / dimensions[0]
                        vec2 = cell[1] * (dimensions[1] - 1) / dimensions[1]
                        vec3 = cell[2] * (dimensions[2] - 1) / dimensions[2]

                        for iso in isos:

                            try:

                                if args.no_clip == True:
                                    pass
                                else:
                                    iso = iso.clip_surface(
                                        bz_surf, invert=True
                                    )
                                plotter.add_mesh(
                                    iso,
                                    lighting=True,
                                    color=color_list[2 * counter],
                                    opacity=args.opacity,
                                    backface_params={
                                        "color": color_list[2 * counter + 1]
                                    },
                                )

                            except:
                                # print(file + " contains an empty mesh")
                                pass

                            conversion_factor = 0.52917721067  # Conversion factor for units (Bohr to Angstroms)

                            orbit_line1 = pv.MultipleLines(
                                points=np.array(
                                    [
                                        conversion_factor
                                        * loop1[:, 0]
                                        / (2 * np.pi * 10e9),
                                        conversion_factor
                                        * loop1[:, 1]
                                        / (2 * np.pi * 10e9),
                                        conversion_factor
                                        * loop1[:, 2]
                                        / (2 * np.pi * 10e9),
                                    ]
                                ).T
                            )
                            plotter.add_mesh(
                                orbit_line1,
                                color=args.loop1_colour,
                                line_width=args.resolution
                                * args.loop_linewidth,
                            )

                            orbit_line2 = pv.MultipleLines(
                                points=np.array(
                                    [
                                        conversion_factor
                                        * loop2[:, 0]
                                        / (2 * np.pi * 10e9),
                                        conversion_factor
                                        * loop2[:, 1]
                                        / (2 * np.pi * 10e9),
                                        conversion_factor
                                        * loop2[:, 2]
                                        / (2 * np.pi * 10e9),
                                    ]
                                ).T
                            )
                            plotter.add_mesh(
                                orbit_line2,
                                color=args.loop2_colour,
                                line_width=args.resolution
                                * args.loop_linewidth,
                            )

                            # # Highlight the closest points
                            p1, p2 = points
                            connect_line = pv.Line(
                                conversion_factor * p1 / (2 * np.pi * 10e9),
                                conversion_factor * p2 / (2 * np.pi * 10e9),
                            )
                            plotter.add_mesh(
                                connect_line,
                                color=args.connect_colour,
                                line_width=args.resolution
                                * args.connect_linewidth,
                            )

                            counter += 1

                    # plot BZ
                    for xx in e:
                        line = pv.MultipleLines(
                            points=np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T
                        )
                        plotter.add_mesh(line, color="black", line_width=2 * 2)

                    plotter.set_background("white")
                    plotter.camera_position = "yz"
                    plotter.set_position([0.5 / args.zoom, 0, 0])
                    plotter.camera.azimuth = args.azimuth
                    plotter.camera.elevation = args.elevation
                    # Save overall plot
                    plotter.save_graphic(
                        f"FS_bands_{band_list[0]}_{band_list[1]}_slices_{slice_comp[0][counter1]}_{slice_comp[1][counter2]}.pdf"
                    )
                    plotter.show()

            counter2 += 1
        counter1 += 1

    # plt.show()


if __name__ == "__main__":
    main()
