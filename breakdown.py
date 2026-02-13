#!/usr/bin/env python3
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

# --- Physical Constants & Unit Conversions ---
# 1 Bohr (a0) = 5.29177210903e-11 meters
# 1 Angstrom = 1e-10 meters
BOHR_M = ct.physical_constants['Bohr radius'][0]
ANGSTROM_M = 1e-10

# Inverse conversions (Length^-1 -> m^-1)
INV_BOHR_TO_INV_M = 1 / BOHR_M  # ~ 1.8897e10 m^-1
INV_ANG_TO_INV_M = 1 / ANGSTROM_M # = 1e10 m^-1

def args_parser_breakdown() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakdown Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # File & Band Selection
    parser.add_argument("-n", "--name", type=str, required=True, help="bxsf file name")
    parser.add_argument("-b", "--bands", type=str, required=True, help="List of bands (e.g., '[10, 11]')")
    
    # Physics / Units
    parser.add_argument("-s", "--spin", choices=["u", "d", "ud", "n"], default="n")
    parser.add_argument("-u", "--units", choices=["au", "ang"], default="au", 
                        help="Input units of the bxsf/skeaf files. 'au' = Inverse Bohr, 'ang' = Inverse Angstrom.")
    parser.add_argument("-i", "--interpolation", type=int, default=3, help="Spline interpolation degree")
    parser.add_argument("-se", "--shift-energy", type=float, default=0.0)
    parser.add_argument("-sk-t", "--skeaf-theta", type=float, default=0.0)
    parser.add_argument("-sk-p", "--skeaf-phi", type=float, default=0.0)

    # Plotting
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
    """Closes and smoothes the orbit loop using cubic splines."""
    if len(loop_points) < 3:
        return loop_points
    t = np.arange(len(loop_points))
    t_fine = np.linspace(0, len(loop_points) - 1, factor * len(loop_points))
    cs = CubicSpline(t, loop_points, bc_type="periodic")
    fine_points = cs(t_fine)
    return np.vstack([fine_points, fine_points[0]])

def read_loops(file_path: str, interpol: int, unit_scale_factor: float) -> tuple[list[np.ndarray], list[str]]:
    """
    Reads SKEAF output and converts coords to m^-1.
    unit_scale_factor: Multiplier to go from Input Units -> m^-1.
    """
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
                    # Apply unit conversion immediately
                    coords = [float(c) * unit_scale_factor for c in clean.split()]
                    current_loop.append(coords)
                except ValueError:
                    continue

    if current_loop:
        loops.append(interpolate_loop(np.array(current_loop), interpol))
        slices.append(current_slice)
            
    return loops, slices

def calculate_curvature_radius(loop: np.ndarray, point: np.ndarray) -> float:
    """
    Calculates Radius of Curvature (rho) in k-space.
    Units of rho are [Length^-1] (same as k-space).
    """
    idx = np.argmin(np.linalg.norm(loop - point, axis=1))
    t = np.arange(len(loop))
    cs = CubicSpline(t, loop, bc_type="periodic")
    
    r1 = cs(idx, 1) # First derivative (tangent)
    r2 = cs(idx, 2) # Second derivative
    
    mag_r1 = np.linalg.norm(r1)
    mag_cross = np.linalg.norm(np.cross(r1, r2))
    
    if mag_cross == 0:
        return float("inf")
    return (mag_r1**3) / mag_cross

def main():
    args = args_parser_breakdown().parse_args()
    bxsf_files = load_files(args)
    
    if not bxsf_files:
        print("Error: No .bxsf files found.")
        return

    # --- 1. Unit Setup ---
    # Determine scale factor based on input units
    if args.units == "au":
        scale_factor = INV_BOHR_TO_INV_M
        unit_name = "Inverse Bohr"
    else:
        scale_factor = INV_ANG_TO_INV_M
        unit_name = "Inverse Angstrom"

    # --- 2. Load Lattice Vectors ---
    _, _, _, _, _, _, raw_cell = read_bxsf_info(bxsf_files[0])
    
    # Convert Lattice Vectors to m^-1
    # raw_cell rows are typically b1, b2, b3
    recip_lattice_m = np.array(raw_cell) * scale_factor
    
    # Calculate Inverse Basis for Fractional Mapping (m^-1 -> Fractional)
    try:
        recip_basis_inv = np.linalg.inv(recip_lattice_m)
    except np.linalg.LinAlgError:
        print("Error: Singular lattice matrix.")
        return

    # Prepare BZ Surface for plotting (Visuals use BZ scale)
    v, e, _ = get_brillouin_zone_3d(raw_cell) # Keep BZ in original units for PyVista scaling later
    bz_surf = pv.PolyData(v).delaunay_3d().extract_surface()

    # --- 3. Load Orbits ---
    band_list = ast.literal_eval(args.bands)
    if len(band_list) == 1: band_list *= 2
    
    all_loops, all_slices = [], []
    for b in band_list:
        pattern = f"./skeaf_out/*band-{b}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}*"
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"Error: No SKEAF output for band {b}")
            return
        # Pass scale_factor to convert everything to m^-1 immediately
        l, s = read_loops(matching_files[0], args.interpolation, scale_factor)
        all_loops.append(l)
        all_slices.append(s)

    # --- 4. Analysis Loop ---
    print(f"\n{'='*90}")
    print(f"MAGNETIC BREAKDOWN ANALYSIS")
    print(f"Input Units: {unit_name} -> Converted to m^-1")
    print(f"Recip. Lattice Vectors (m^-1):")
    for row in recip_lattice_m:
        print(f"  {row}")
    print(f"{'='*90}")

    for i, loop1 in enumerate(all_loops[0]):
        for j, loop2 in enumerate(all_loops[1]):
            # A. Find Closest Points
            dists = distance_matrix(loop1, loop2)
            idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
            p1, p2 = loop1[idx1], loop2[idx2]
            
            # B. Gap Geometry
            gap_vec_cart = p1 - p2             # Vector in m^-1
            dist_abs = np.linalg.norm(gap_vec_cart)
            
            # Project cartesian gap vector onto reciprocal basis to get fractional components
            # gap_frac = gap_cart * B^-1
            gap_vec_frac = np.dot(gap_vec_cart, recip_basis_inv)

            # C. Mean Kz Position
            mean_kz_abs = (np.mean(loop1[:, 2]) + np.mean(loop2[:, 2])) / 2
            # Project mean position vector [0, 0, kz] to fractional
            mean_kz_frac = np.dot(np.array([0, 0, mean_kz_abs]), recip_basis_inv)[2]

            # D. Physics (Curvature & Breakdown)
            rho1 = calculate_curvature_radius(loop1, p1)
            rho2 = calculate_curvature_radius(loop2, p2)
            
            b0_str = "Undefined"
            if not (np.isinf(rho1) or np.isinf(rho2)):
                # Pippard Formula: B0 = (pi*hbar)/(2e) * sqrt( k_g^3 / (1/rho1 + 1/rho2) )
                # All units are SI (m, m^-1)
                prefactor = (np.pi * ct.hbar) / (2 * ct.e)
                curvature_sum = (1.0/rho1) + (1.0/rho2)
                b0 = prefactor * np.sqrt( (dist_abs**3) / curvature_sum )
                b0_str = f"{b0:.2f} T"

            # E. Output
            print(f"\n--- Band {band_list[0]} (Slice {all_slices[0][i]}) vs Band {band_list[1]} (Slice {all_slices[1][j]}) ---")
            print(f"Mean kz Position:           {mean_kz_abs:.4e} m^-1  (Fractional kz: {mean_kz_frac:.4f})")
            print(f"Gap Distance (Absolute):    {dist_abs:.4e} m^-1")
            print(f"Gap Vector (Fractional):    [{gap_vec_frac[0]:.4f}, {gap_vec_frac[1]:.4f}, {gap_vec_frac[2]:.4f}]")
            print(f"Radii of Curvature (k-space):")
            print(f"  - Orbit 1 (Band {band_list[0]}): {rho1:.4e} m^-1")
            print(f"  - Orbit 2 (Band {band_list[1]}): {rho2:.4e} m^-1")
            print(f"Critical Field B0:          {b0_str}")

            # F. Plotting (if requested)
            if args.plot:
                # Viz Scale: Shrink m^-1 back to Angstrom-ish size for PyVista rendering stability
                viz_scale = 1.0 / INV_ANG_TO_INV_M 
                
                plotter = pv.Plotter(window_size=[int(args.resolution * 1024), int(args.resolution * 768)])
                colors = sns.color_palette("hls", 2 * len(bxsf_files))

                # Plot Fermi Surface (convert raw units -> Viz Scale)
                # Note: read_bxsf returns isos in raw units (e.g. Inv Bohr). We must match scaling.
                fs_scale_factor = scale_factor * viz_scale
                
                for idx, file in enumerate(bxsf_files):
                    _, _, _, _, _, isos, _ = read_bxsf(
                        file, args.plot_interpolate, order=args.plot_order, shift_energy=args.shift_energy
                    )
                    for iso in isos:
                        try:
                            # Manually scale the mesh points
                            iso.points *= fs_scale_factor
                            
                            # Clip (Bz surf must also be scaled)
                            bz_surf_scaled = bz_surf.copy()
                            bz_surf_scaled.points *= fs_scale_factor
                            
                            mesh = iso if args.no_clip else iso.clip_surface(bz_surf_scaled, invert=True)
                            plotter.add_mesh(mesh, color=colors[2*idx], opacity=args.opacity, lighting=True)
                        except: continue

                # Plot Loops & Connections
                plotter.add_mesh(pv.MultipleLines(points=loop1 * viz_scale), color=args.loop1_colour, line_width=5)
                plotter.add_mesh(pv.MultipleLines(points=loop2 * viz_scale), color=args.loop2_colour, line_width=5)
                plotter.add_mesh(pv.Line(p1 * viz_scale, p2 * viz_scale), color=args.connect_colour, line_width=5)

                # Plot BZ Edges
                for edge in e:
                    plotter.add_mesh(pv.MultipleLines(points=edge * fs_scale_factor), color="black", line_width=2)

                plotter.set_background("white")
                plotter.camera.azimuth, plotter.camera.elevation = args.azimuth, args.elevation
                plotter.show()

if __name__ == "__main__":
    main()