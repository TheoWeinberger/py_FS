#! /usr/bin/env python3
import argparse
import ast
import glob
import numpy as np
import pyvista as pv
import seaborn as sns
import scipy.constants as ct
from scipy.interpolate import CubicSpline
from scipy.spatial import distance_matrix
from src.utils import (
    get_brillouin_zone_3d,
    read_bxsf_info,
    load_files,
    read_bxsf,
)

# Constants
BOHR_TO_ANGSTROM = 0.52917721067
CONVERSION_TO_INV_M = 1e10

def args_parser_breakdown() -> argparse.ArgumentParser:
    """
    Function to take input command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Breakdown Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Logic Arguments
    parser.add_argument("-n", "--name", type=str, required=True, help="bxsf file name")
    parser.add_argument("-b", "--bands", type=str, required=True, help="List of bands (e.g., '[10, 11]')")
    parser.add_argument("-s", "--spin", choices=["u", "d", "ud", "n"], default="n")
    parser.add_argument("-i", "--interpolation", type=int, default=3, help="Spline interpolation degree")
    parser.add_argument("-u", "--units", choices=["au", "ang"], default="au")
    parser.add_argument("-se", "--shift-energy", type=float, default=0.0)
    parser.add_argument("-sk-t", "--skeaf-theta", type=float, default=0.0)
    parser.add_argument("-sk-p", "--skeaf-phi", type=float, default=0.0)

    # Plotting Arguments
    parser.add_argument("-p", "--plot", type=bool, default=False)
    parser.add_argument("-pi", "--plot-interpolate", type=int, default=1)
    parser.add_argument("-po", "--plot-order", type=int, default=1)
    parser.add_argument("-op", "--opacity", type=float, default=0.2)
    parser.add_argument("-r", "--resolution", type=float, default=1.0)
    parser.add_argument("-l1c", "--loop1-colour", default="red")
    parser.add_argument("-l2c", "--loop2-colour", default="blue")
    parser.add_argument("-ll", "--loop-linewidth", type=float, default=8.0)
    parser.add_argument("-cc", "--connect-colour", default="black")
    parser.add_argument("-cl", "--connect-linewidth", type=float, default=5.0)
    parser.add_argument("-nc", "--no-clip", type=bool, default=False)
    parser.add_argument("-a", "--azimuth", type=float, default=65)
    parser.add_argument("-e", "--elevation", type=float, default=30)
    parser.add_argument("-z", "--zoom", type=float, default=1.0)
    
    return parser

def interpolate_loop(loop_points: np.ndarray, factor: int) -> np.ndarray:
    """Applies cubic spline interpolation to close and smooth the orbits."""
    if len(loop_points) < 3:
        return loop_points
    t = np.arange(len(loop_points))
    t_fine = np.linspace(0, len(loop_points) - 1, factor * len(loop_points))
    
    cs = CubicSpline(t, loop_points, bc_type="periodic")
    fine_points = cs(t_fine)
    return np.vstack([fine_points, fine_points[0]])

def read_loops(file_path: str, interpol: int, units: str = "au") -> tuple[list[np.ndarray], list[str]]:
    """Reads SKEAF orbit output and converts units."""
    conv = BOHR_TO_ANGSTROM if units == "au" else 1.0
    loops, slices = [], []
    current_loop = []
    current_slice = "Unknown"

    with open(file_path, "r") as f:
        for line in f:
            clean = line.strip()
            if clean.startswith("Slice"):
                current_slice = clean.split("=")[1].split(",")[0].strip()
            elif clean.startswith("Points"):
                if current_loop:
                    loops.append(interpolate_loop(np.array(current_loop), interpol))
                    slices.append(current_slice)
                    current_loop = []
            elif len(clean.split()) == 3:
                try:
                    # Convert to inverse meters
                    coords = [float(c) * CONVERSION_TO_INV_M / conv for c in clean.split()]
                    current_loop.append(coords)
                except ValueError:
                    continue

    if current_loop:
        loops.append(interpolate_loop(np.array(current_loop), interpol))
        slices.append(current_slice)
            
    return loops, slices

def calculate_radius_of_curvature(loop: np.ndarray, point: np.ndarray) -> float:
    """Calculates local radius of curvature at the point of closest approach."""
    idx = np.argmin(np.linalg.norm(loop - point, axis=1))
    t = np.arange(len(loop))
    cs = CubicSpline(t, loop, bc_type="periodic")
    
    r1 = cs(idx, 1) # First derivative
    r2 = cs(idx, 2) # Second derivative
    
    mag_r1 = np.linalg.norm(r1)
    mag_cross = np.linalg.norm(np.cross(r1, r2))
    
    return (mag_r1**3) / mag_cross if mag_cross != 0 else float("inf")

def main():
    args = args_parser_breakdown().parse_args()
    bxsf_files = load_files(args)
    
    if not bxsf_files:
        print("Error: No .bxsf files found. Check --name argument.")
        return

    # Process Brillouin Zone for clipping and plotting
    _, _, _, _, _, _, cell = read_bxsf_info(bxsf_files[0])
    v, e, _ = get_brillouin_zone_3d(cell)
    bz_surf = pv.PolyData(v).delaunay_3d().extract_surface()

    # Parse band selection
    band_list = ast.literal_eval(args.bands)
    if len(band_list) == 1: band_list *= 2
    
    # Load orbit data from SKEAF output files
    all_loops, all_slices = [], []
    for b in band_list:
        pattern = f"./skeaf_out/*band-{b}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}*"
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"Error: Could not find SKEAF output for band {b}")
            return
        l, s = read_loops(matching_files[0], args.interpolation, args.units)
        all_loops.append(l)
        all_slices.append(s)

    # Unit scaling for PyVista visualization (Inverse Meters -> Viz Scale)
    viz_scale = BOHR_TO_ANGSTROM / (2 * np.pi * 1e10)

    # Compare every slice from Band A to every slice from Band B
    for i, loop1 in enumerate(all_loops[0]):
        for j, loop2 in enumerate(all_loops[1]):
            # 1. Find the gap (distance of closest approach)
            dists = distance_matrix(loop1, loop2)
            idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
            p1, p2 = loop1[idx1], loop2[idx2]
            dist = dists[idx1, idx2]

            # 2. Calculate Curvature and Breakdown Field B0
            rho1 = calculate_radius_of_curvature(loop1, p1)
            rho2 = calculate_radius_of_curvature(loop2, p2)
            
            b0_label = "Undefined (Infinite Radius)"
            if not (np.isinf(rho1) or np.isinf(rho2)):
                b0 = (np.pi / 2) * (ct.hbar / ct.e) * np.sqrt((dist**3) / (1/rho1 + 1/rho2))
                b0_label = f"{b0:.2f} T"

            print(f"\n--- Band {band_list[0]} ({all_slices[0][i]}) vs Band {band_list[1]} ({all_slices[1][j]}) ---")
            print(f"Gap Distance: {dist:.4e} 1/m")
            print(f"Magnetic Breakdown Field B0: {b0_label}")

            # 3. 3D Visualization
            if args.plot:
                plotter = pv.Plotter(window_size=[int(args.resolution * 1024), int(args.resolution * 768)])
                colors = sns.color_palette("hls", 2 * len(bxsf_files))

                for idx, file in enumerate(bxsf_files):
                    _, _, _, _, _, isos, _ = read_bxsf(
                        file, args.plot_interpolate, order=args.plot_order, shift_energy=args.shift_energy
                    )
                    for iso in isos:
                        try:
                            mesh = iso if args.no_clip else iso.clip_surface(bz_surf, invert=True)
                            plotter.add_mesh(mesh, color=colors[2*idx], opacity=args.opacity, lighting=True)
                        except: continue

                # Draw Orbits and Connection
                plotter.add_mesh(pv.MultipleLines(points=loop1 * viz_scale), color=args.loop1_colour, line_width=args.resolution * args.loop_linewidth)
                plotter.add_mesh(pv.MultipleLines(points=loop2 * viz_scale), color=args.loop2_colour, line_width=args.resolution * args.loop_linewidth)
                plotter.add_mesh(pv.Line(p1 * viz_scale, p2 * viz_scale), color=args.connect_colour, line_width=args.resolution * args.connect_linewidth)

                # Draw BZ Edges
                for edge in e:
                    plotter.add_mesh(pv.MultipleLines(points=edge), color="black", line_width=2)

                plotter.set_background("white")
                plotter.camera.azimuth, plotter.camera.elevation = args.azimuth, args.elevation
                plotter.show()

if __name__ == "__main__":
    main()