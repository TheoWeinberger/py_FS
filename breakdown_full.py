#!/usr/bin/env python3
import argparse
import ast
import glob
import numpy as np
import pyvista as pv
import seaborn as sns
import scipy.constants as ct
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.spatial import distance_matrix

from src.utils import (
    read_bxsf_info,
    read_bxsf,
    get_brillouin_zone_3d,
)

# Physical Constants & Unit Conversions
BOHR_M = ct.physical_constants['Bohr radius'][0]
ANGSTROM_M = 1e-10
INV_BOHR_TO_INV_M = 1 / BOHR_M
INV_ANG_TO_INV_M = 1 / ANGSTROM_M

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def interpolate_loop(loop_points: np.ndarray, factor: int) -> np.ndarray:
    """Smoothes the orbit loop using parametric splines designed for closed curves."""
    if len(loop_points) < 3:
        return loop_points
    # Use splprep for parametric spline fitting on closed curves
    # s=0 means exact fit, per=1 means periodic (closed curve)
    tck, u = splprep(loop_points.T, s=0, per=True, k=min(3, len(loop_points) - 1))
    # Evaluate at finer resolution without the endpoint (to avoid duplication)
    u_fine = np.linspace(0, 1, factor * len(loop_points), endpoint=False)
    fine_points = np.column_stack(splev(u_fine, tck))
    # Explicitly close the loop by appending the first point
    fine_points = np.vstack([fine_points, fine_points[0]])
    return fine_points

def calculate_curvature_radius(loop: np.ndarray, point: np.ndarray) -> float:
    """Calculate radius of curvature at closest point on loop"""
    idx = np.argmin(np.linalg.norm(loop - point, axis=1))
    # Use splprep for parametric spline fitting on closed curves
    tck, u = splprep(loop.T, s=0, per=True, k=min(3, len(loop) - 1))
    
    # Evaluate derivatives at the closest point
    r1 = np.array(splev(u[idx], tck, der=1))  # First derivative (tangent)
    r2 = np.array(splev(u[idx], tck, der=2))  # Second derivative
    
    mag_r1 = np.linalg.norm(r1)
    mag_cross = np.linalg.norm(np.cross(r1, r2))
    
    if mag_cross < 1e-10:
        return float("inf")
    return (mag_r1**3) / mag_cross

def find_connected_components(points: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Simple connected components clustering without sklearn."""
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    
    for i in range(n):
        if visited[i]:
            continue
        
        stack = [i]
        labels[i] = cluster_id
        visited[i] = True
        
        while stack:
            current = stack.pop()
            dists = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where((dists < eps) & (~visited))[0]
            
            for neighbor in neighbors:
                visited[neighbor] = True
                labels[neighbor] = cluster_id
                stack.append(neighbor)
        
        cluster_id += 1
    
    return labels

def sort_loop_points(points: np.ndarray) -> np.ndarray:
    """Sort intersection points to form a continuous closed loop"""
    if len(points) < 3:
        return points
    
    loop = [points[0]]
    remaining = list(range(1, len(points)))
    
    while remaining:
        last = loop[-1]
        dists = np.linalg.norm(points[remaining] - last, axis=1)
        nearest_idx = np.argmin(dists)
        loop.append(points[remaining[nearest_idx]])
        remaining.pop(nearest_idx)
    
    return np.array(loop)

def clip_to_first_bz(loop_points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Filter loop points to keep only those inside the first Brillouin zone"""
    if len(loop_points) == 0:
        return loop_points
    
    # Generate reciprocal lattice points
    extent = 1
    lattice_points_list = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                lattice_pt = cell[0] * i + cell[1] * j + cell[2] * k
                lattice_points_list.append(lattice_pt)
    
    lattice_points = np.array(lattice_points_list)
    
    inside_mask = np.ones(len(loop_points), dtype=bool)
    
    for i, point in enumerate(loop_points):
        # Distance to Gamma (origin)
        dist_origin = np.linalg.norm(point)
        
        # Check if point is closer to Gamma than to any other lattice point
        for lattice_pt in lattice_points:
            if np.allclose(lattice_pt, [0, 0, 0]):
                continue
            dist_to_lattice = np.linalg.norm(point - lattice_pt)
            if dist_to_lattice < dist_origin - 1e-8:
                inside_mask[i] = False
                break
    
    clipped_points = loop_points[inside_mask]
    return clipped_points

def plane_intersection_with_isosurface(isosurface: pv.PolyData, 
                                       plane_normal: np.ndarray, 
                                       plane_point: np.ndarray) -> np.ndarray:
    """Find intersection points between plane and isosurface mesh"""
    normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    vertices = isosurface.points
    distances = np.dot(vertices - plane_point, normal)
    
    intersection_points = []
    
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    print(f"  Vertices with distance < 1e-6: {np.sum(np.abs(distances) < 1e-6)}")
    print(f"  Vertices with distance < 0: {np.sum(distances < 0)}")
    print(f"  Vertices with distance > 0: {np.sum(distances > 0)}")
    
    # Iterate over all cells
    for cell_id in range(isosurface.n_cells):
        cell = isosurface.get_cell(cell_id)
        cell_points = cell.points
        
        # Check all edges of the cell (triangle)
        for i in range(len(cell_points)):
            v1_idx = cell.point_ids[i]
            v2_idx = cell.point_ids[(i + 1) % len(cell.point_ids)]
            
            d1 = distances[v1_idx]
            d2 = distances[v2_idx]
            
            # Check if edge crosses plane
            if d1 * d2 < 1e-10 and not (abs(d1) < 1e-10 and abs(d2) < 1e-10):
                v1, v2 = vertices[v1_idx], vertices[v2_idx]
                if abs(d1) < 1e-10:
                    intersection = v1
                elif abs(d2) < 1e-10:
                    intersection = v2
                else:
                    t = d1 / (d1 - d2)
                    intersection = v1 + t * (v2 - v1)
                intersection_points.append(intersection)
    
    if not intersection_points:
        print(f"  Warning: No intersection found between plane and isosurface")
        return np.array([])
    
    intersection_points = np.array(intersection_points)
    print(f"  Found {len(intersection_points)} raw intersection points")
    
    # Remove duplicates
    tolerance = 1e-6
    unique_points = []
    for pt in intersection_points:
        if not unique_points or np.min(np.linalg.norm(np.array(unique_points) - pt, axis=1)) > tolerance:
            unique_points.append(pt)
    
    intersection_points = np.array(unique_points)
    print(f"  After deduplication: {len(intersection_points)} points")
    
    # Cluster points to identify separate loops
    if len(intersection_points) > 10:
        labels = find_connected_components(intersection_points, eps=0.01)
        
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        largest_label = unique_labels[np.argmax(cluster_sizes)]
        
        intersection_points = intersection_points[labels == largest_label]
        print(f"  Selected main loop with {len(intersection_points)} points (largest of {len(unique_labels)} clusters)")
    
    loop = sort_loop_points(intersection_points)
    
    return loop

def find_all_orbit_plane_intersections(loop: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> list:
    """Find all intersection points of an orbit (closed loop) with a plane"""
    normal = plane_normal / np.linalg.norm(plane_normal)
    
    distances = np.dot(loop - plane_point, normal)
    
    print(f"      Loop points: {len(loop)}")
    print(f"      Distance range from plane: [{distances.min():.6f}, {distances.max():.6f}]")
    
    crossings = []
    
    for i in range(len(loop) - 1):
        d1, d2 = distances[i], distances[i+1]
        
        if d1 * d2 < 0:
            t = d1 / (d1 - d2)
            crossing_pt = loop[i] + t * (loop[i+1] - loop[i])
            crossings.append(crossing_pt)
        elif np.abs(d1) < 1e-8:
            crossings.append(loop[i])
    
    if len(distances) > 1:
        d_last = distances[-1]
        d_first = distances[0]
        if d_last * d_first < 0:
            t = d_last / (d_last - d_first)
            crossing_pt = loop[-1] + t * (loop[0] - loop[-1])
            crossings.append(crossing_pt)
    
    unique_crossings = []
    tol = 1e-6
    for pt in crossings:
        is_duplicate = False
        for existing in unique_crossings:
            if np.linalg.norm(pt - existing) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_crossings.append(pt)
    
    print(f"      Found {len(unique_crossings)} unique crossings")
    return unique_crossings

def read_loops_from_skeaf(file_path: str, interpol: int, unit_scale_factor: float) -> tuple[list[np.ndarray], list[str]]:
    """Reads SKEAF output and converts coords to SI (m^-1)."""
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
                    coords = [float(c) * unit_scale_factor for c in clean.split()]
                    current_loop.append(coords)
                except ValueError:
                    continue

    if current_loop:
        loops.append(interpolate_loop(np.array(current_loop), interpol))
        slices.append(current_slice)
            
    return loops, slices

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakdown Analysis Tool - Combine SKEAF or Plane-based Fermi Surface Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument("-m", "--mode", choices=["skeaf", "plane"], required=True,
                        help="Analysis mode: 'skeaf' for SKEAF output or 'plane' for plane intersection")
    
    # File & Band Selection (common)
    parser.add_argument("-n", "--name", type=str, required=True, help="bxsf file name prefix")
    
    # Bands (common to both modes)
    parser.add_argument("-b", "--bands", type=str, default=None,
                        help="List of bands (e.g., '[10, 11]')")
    
    # SKEAF mode arguments
    parser.add_argument("-sk-t", "--skeaf-theta", type=float, default=0.0,
                        help="SKEAF theta parameter")
    parser.add_argument("-sk-p", "--skeaf-phi", type=float, default=0.0,
                        help="SKEAF phi parameter")
    
    # Plane Definition (plane mode)
    parser.add_argument("-pn", "--plane-normal", type=float, nargs=3, default=[0, 0, 1],
                        help="Plane normal vector [nx, ny, nz]")
    parser.add_argument("-pp", "--plane-point", type=float, nargs=3, default=[0, 0, 0],
                        help="Point on plane [x, y, z]")
    parser.add_argument("-pn2", "--plane-normal-2", type=float, nargs=3, default=None,
                        help="Second perpendicular plane normal vector for breakdown point selection")
    parser.add_argument("-pp2", "--plane-point-2", type=float, nargs=3, default=[0, 0, 0],
                        help="Point on second plane [x, y, z]")
    
    # Physics / Units (common)
    parser.add_argument("-u", "--units", choices=["au", "ang"], default="au",
                        help="Input units: 'au' = Inverse Bohr, 'ang' = Inverse Angstrom")
    parser.add_argument("-i", "--interpolation", type=int, default=3,
                        help="Spline interpolation degree")
    parser.add_argument("-se", "--shift-energy", type=float, default=0.0,
                        help="Energy shift (plane mode)")
    parser.add_argument("-scale", type=int, default=1, help="Grid interpolation scale factor (plane mode)")
    parser.add_argument("-order", type=int, default=1, help="Interpolation order (plane mode)")
    
    # Spin (SKEAF mode)
    parser.add_argument("-s", "--spin", choices=["u", "d", "ud", "n"], default="n",
                        help="Spin polarization for SKEAF mode")
    
    # Plotting (common)
    parser.add_argument("-p", "--plot", type=bool, default=False,
                        help="Generate 3D visualization")
    parser.add_argument("-op", "--opacity", type=float, default=0.2,
                        help="Opacity for surface plots")
    parser.add_argument("-a", "--azimuth", type=float, default=65,
                        help="Camera azimuth angle")
    parser.add_argument("-e", "--elevation", type=float, default=30,
                        help="Camera elevation angle")
    
    # Additional plotting options (SKEAF mode)
    parser.add_argument("-pi", "--plot-interpolate", type=int, default=1,
                        help="Interpolation factor for SKEAF plotting")
    parser.add_argument("-po", "--plot-order", type=int, default=1,
                        help="Plot interpolation order (SKEAF mode)")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                        help="Plot resolution scale")
    parser.add_argument("-l1c", "--loop1-colour", default="red",
                        help="Color for loop 1")
    parser.add_argument("-l2c", "--loop2-colour", default="blue",
                        help="Color for loop 2")
    parser.add_argument("-ll", "--loop-linewidth", type=float, default=8.0,
                        help="Loop line width")
    parser.add_argument("-cc", "--connect-colour", default="black",
                        help="Color for connecting line")
    parser.add_argument("-cl", "--connect-linewidth", type=float, default=5.0,
                        help="Connecting line width")
    parser.add_argument("-nc", "--no-clip", type=bool, default=False,
                        help="Don't clip to BZ")
    parser.add_argument("-z", "--zoom", type=float, default=1.0,
                        help="Camera zoom factor")
    
    return parser

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def load_bxsf_files(name: str, bands: list, spin: str) -> list[str]:
    """Load bxsf files based on band list and spin."""
    files = []
    for band in bands:
        if spin == "u":
            files += glob.glob(name + f"*bxsf.band-{band}_up")
        elif spin == "d":
            files += glob.glob(name + f"*bxsf.band-{band}_down")
        elif spin == "ud":
            files += glob.glob(name + f"*bxsf.band-{band}_*")
        elif spin == "n":
            files += glob.glob(name + f"*bxsf.band-{band}")
    
    return files

def analyze_skeaf_mode(args):
    """Analysis using SKEAF output orbits"""
    print(f"\n{'='*90}")
    print(f"SKEAF-BASED MAGNETIC BREAKDOWN ANALYSIS")
    print(f"{'='*90}\n")
    
    # Unit setup
    if args.units == "au":
        scale_factor = INV_BOHR_TO_INV_M
        unit_name = "Inverse Bohr"
    else:
        scale_factor = INV_ANG_TO_INV_M
        unit_name = "Inverse Angstrom"
    
    # Load lattice vectors
    bxsf_files = load_bxsf_files(args.name, ast.literal_eval(args.bands), args.spin)
    
    if not bxsf_files:
        print(f"Error: No bxsf files found matching pattern")
        return
    
    _, _, _, _, _, _, raw_cell = read_bxsf_info(bxsf_files[0])
    recip_lattice_m = np.array(raw_cell) * scale_factor
    
    try:
        recip_basis_inv = np.linalg.inv(recip_lattice_m)
    except np.linalg.LinAlgError:
        print("Error: Singular lattice matrix.")
        return
    
    # Prepare BZ
    v, e, _ = get_brillouin_zone_3d(raw_cell)
    bz_surf = pv.PolyData(v).delaunay_3d().extract_surface()
    
    # Load orbits from SKEAF output
    band_list = ast.literal_eval(args.bands)
    if len(band_list) == 1:
        band_list *= 2
    
    all_loops, all_slices = [], []
    for b in band_list:
        pattern = f"./skeaf_out/*band-{b}_theta_{args.skeaf_theta}_phi_{args.skeaf_phi}*"
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"Error: No SKEAF output for band {b}")
            return
        l, s = read_loops_from_skeaf(matching_files[0], args.interpolation, scale_factor)
        all_loops.append(l)
        all_slices.append(s)
    
    print(f"Input Units: {unit_name} -> Converted to m^-1")
    print(f"Recip. Lattice Vectors (m^-1):")
    for row in recip_lattice_m:
        print(f"  {row}")
    
    # Analysis loop
    for i, loop1 in enumerate(all_loops[0]):
        for j, loop2 in enumerate(all_loops[1]):
            # Find closest points
            dists = distance_matrix(loop1, loop2)
            idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
            p1, p2 = loop1[idx1], loop2[idx2]
            
            # Gap geometry
            gap_vec_cart = p1 - p2
            dist_abs = np.linalg.norm(gap_vec_cart)
            gap_vec_frac = np.dot(gap_vec_cart, recip_basis_inv)
            
            # Mean Kz position
            mean_kz_abs = (np.mean(loop1[:, 2]) + np.mean(loop2[:, 2])) / 2
            mean_kz_frac = np.dot(np.array([0, 0, mean_kz_abs]), recip_basis_inv)[2]
            
            # Physics
            rho1 = calculate_curvature_radius(loop1, p1)
            rho2 = calculate_curvature_radius(loop2, p2)
            
            b0_str = "Undefined"
            if not (np.isinf(rho1) or np.isinf(rho2)):
                prefactor = (np.pi * ct.hbar) / (2 * ct.e)
                curvature_sum = (1.0/rho1) + (1.0/rho2)
                b0 = prefactor * np.sqrt((dist_abs**3) / curvature_sum)
                b0_str = f"{b0:.2f} T"
            
            # Output
            print(f"\n--- Band {band_list[0]} (Slice {all_slices[0][i]}) vs Band {band_list[1]} (Slice {all_slices[1][j]}) ---")
            print(f"Mean kz Position:           {mean_kz_abs:.4e} m^-1  (Fractional kz: {mean_kz_frac:.4f})")
            print(f"Gap Distance (Absolute):    {dist_abs:.4e} m^-1")
            print(f"Gap Vector (Fractional):    [{gap_vec_frac[0]:.4f}, {gap_vec_frac[1]:.4f}, {gap_vec_frac[2]:.4f}]")
            print(f"Radii of Curvature (k-space):")
            print(f"  - Orbit 1 (Band {band_list[0]}): {rho1:.4e} m^-1")
            print(f"  - Orbit 2 (Band {band_list[1]}): {rho2:.4e} m^-1")
            print(f"Critical Field B0:          {b0_str}")
            
            # Plotting
            if args.plot:
                viz_scale = 1.0 / INV_ANG_TO_INV_M
                plotter = pv.Plotter(window_size=[int(args.resolution * 1024), int(args.resolution * 768)])
                colors = sns.color_palette("hls", 2 * len(bxsf_files))
                
                fs_scale_factor = scale_factor * viz_scale
                
                for idx, file in enumerate(bxsf_files):
                    _, _, _, _, _, isos, _ = read_bxsf(
                        file, args.plot_interpolate, order=args.plot_order, shift_energy=args.shift_energy, fermi_velocity=False
                    )
                    for iso in isos:
                        try:
                            iso.points *= fs_scale_factor
                            bz_surf_scaled = bz_surf.copy()
                            bz_surf_scaled.points *= fs_scale_factor
                            mesh = iso if args.no_clip else iso.clip_surface(bz_surf_scaled, invert=True)
                            plotter.add_mesh(mesh, color=colors[2*idx], opacity=args.opacity, lighting=True)
                        except:
                            continue
                
                plotter.add_mesh(pv.MultipleLines(points=loop1 * viz_scale), color=args.loop1_colour, line_width=args.loop_linewidth)
                plotter.add_mesh(pv.MultipleLines(points=loop2 * viz_scale), color=args.loop2_colour, line_width=args.loop_linewidth)
                plotter.add_mesh(pv.Line(p1 * viz_scale, p2 * viz_scale), color=args.connect_colour, line_width=args.connect_linewidth)
                
                for edge in e:
                    plotter.add_mesh(pv.MultipleLines(points=edge * fs_scale_factor), color="black", line_width=2)
                
                plotter.set_background("white")
                plotter.camera.azimuth, plotter.camera.elevation = args.azimuth, args.elevation
                plotter.show()

def analyze_plane_mode(args):
    """Analysis using plane-isosurface intersection"""
    print(f"\n{'='*90}")
    print(f"PLANE-ISOSURFACE BREAKDOWN ANALYSIS")
    print(f"{'='*90}\n")
    
    # Validate plane mode arguments
    if args.bands is None:
        print("Error: Plane mode requires --bands (e.g., '-b [10, 11]')")
        return
    
    try:
        band_list = ast.literal_eval(args.bands)
        if len(band_list) != 2:
            print("Error: Plane mode requires exactly 2 bands")
            return
        band1, band2 = band_list[0], band_list[1]
    except (ValueError, SyntaxError):
        print("Error: Invalid band format. Use -b '[band1, band2]'")
        return
    
    # Unit setup
    if args.units == "au":
        scale_factor = INV_BOHR_TO_INV_M
        unit_name = "Inverse Bohr"
    else:
        scale_factor = INV_ANG_TO_INV_M
        unit_name = "Inverse Angstrom"
    
    # Load BXSF files
    bxsf_file1 = glob.glob(args.name + f"*bxsf.band-{band1}*")
    bxsf_file2 = glob.glob(args.name + f"*bxsf.band-{band2}*")
    
    if not bxsf_file1 or not bxsf_file2:
        print(f"Error: Could not find bxsf files for bands {band1} and {band2}")
        return
    
    bxsf_file1 = bxsf_file1[0]
    bxsf_file2 = bxsf_file2[0]
    print(f"Loading BXSF files:")
    print(f"  Band {band1}: {bxsf_file1}")
    print(f"  Band {band2}: {bxsf_file2}")
    
    # Load reciprocal lattice
    _, _, _, orig_dimensions, _, _, raw_cell = read_bxsf_info(bxsf_file1)
    recip_lattice_m = np.array(raw_cell) * scale_factor
    
    try:
        recip_basis_inv = np.linalg.inv(recip_lattice_m)
    except np.linalg.LinAlgError:
        print("Error: Singular lattice matrix.")
        return
    
    print(f"Input Units: {unit_name} -> Converted to m^-1")
    print(f"Plane Normal: {args.plane_normal}")
    print(f"Plane Point: {args.plane_point}")
    
    # Extract isosurfaces
    try:
        k_vectors1, eig_vals1, e_f1, cell1, dim1, iso1_list, grid1 = read_bxsf(
            bxsf_file1, args.scale, args.order, args.shift_energy, fermi_velocity=False
        )
        iso1 = iso1_list[0]
        
        k_vectors2, eig_vals2, e_f2, cell2, dim2, iso2_list, grid2 = read_bxsf(
            bxsf_file2, args.scale, args.order, args.shift_energy, fermi_velocity=False
        )
        iso2 = iso2_list[0]
    except Exception as e:
        print(f"Error reading BXSF file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get BZ
    bz_vertices, bz_ridges, bz_facets = get_brillouin_zone_3d(raw_cell)
    
    plane_normal = np.array(args.plane_normal)
    plane_point = np.array(args.plane_point)
    plane_point_normalized = plane_point / dim1
    
    print(f"\nFinding plane-isosurface intersections...")
    print("Band 1:")
    loop1 = plane_intersection_with_isosurface(iso1, plane_normal, plane_point_normalized)
    print("Band 2:")
    loop2 = plane_intersection_with_isosurface(iso2, plane_normal, plane_point_normalized)
    
    if len(loop1) < 3 or len(loop2) < 3:
        print("Error: Could not extract valid loops from plane-isosurface intersection")
        return
    
    # Clip to first BZ
    loop1 = clip_to_first_bz(loop1, raw_cell)
    loop2 = clip_to_first_bz(loop2, raw_cell)
    
    if len(loop1) < 3 or len(loop2) < 3:
        print("Error: Not enough points remaining after BZ filtering")
        return
    
    # Interpolate
    loop1 = interpolate_loop(loop1, args.interpolation)
    loop2 = interpolate_loop(loop2, args.interpolation)
    
    # Calculate breakdown
    dists = distance_matrix(loop1, loop2)
    idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
    p1, p2 = loop1[idx1], loop2[idx2]
    
    # If second plane provided, use intersection
    if args.plane_normal_2 is not None:
        print(f"\nUsing second plane for breakdown point selection...")
        plane_normal_2 = np.array(args.plane_normal_2)
        plane_point_2 = np.array(args.plane_point_2)
        
        print(f"  Looking for intersections of orbit 1 with second plane...")
        intersections_1 = find_all_orbit_plane_intersections(loop1, plane_normal_2, plane_point_2)
        
        print(f"  Looking for intersections of orbit 2 with second plane...")
        intersections_2 = find_all_orbit_plane_intersections(loop2, plane_normal_2, plane_point_2)
        
        if intersections_1 and intersections_2:
            pair_dists = distance_matrix(intersections_1, intersections_2)
            i1, i2 = np.unravel_index(np.argmin(pair_dists), pair_dists.shape)
            p1 = intersections_1[i1]
            p2 = intersections_2[i2]
            print(f"  Paired intersection points (distance: {pair_dists[i1, i2]:.6e})")
            
            n1 = plane_normal / np.linalg.norm(plane_normal)
            n2 = plane_normal_2 / np.linalg.norm(plane_normal_2)
            for pt_list in [p1, p2]:
                d1 = np.dot(pt_list - plane_point, n1)
                pt_list[:] = pt_list - d1 * n1
                d2 = np.dot(pt_list - plane_point_2, n2)
                pt_list[:] = pt_list - d2 * n2
        else:
            print(f"  Could not find intersections on both orbits, using closest points instead")
    
    # Convert to SI
    loop1_si = loop1 * scale_factor
    loop2_si = loop2 * scale_factor
    p1_si = p1 * scale_factor
    p2_si = p2 * scale_factor
    
    gap_vec_cart = p1_si - p2_si
    dist_abs = np.linalg.norm(gap_vec_cart)
    gap_vec_frac = np.dot(gap_vec_cart, recip_basis_inv)
    
    rho1 = calculate_curvature_radius(loop1_si, p1_si)
    rho2 = calculate_curvature_radius(loop2_si, p2_si)
    
    b0_str = "Undefined"
    if not (np.isinf(rho1) or np.isinf(rho2)):
        prefactor = (np.pi * ct.hbar) / (2 * ct.e)
        curvature_sum = (1.0/rho1) + (1.0/rho2)
        b0 = prefactor * np.sqrt((dist_abs**3) / curvature_sum)
        b0_str = f"{b0:.2f} T"
    
    print(f"\n--- Band {band1} vs Band {band2} ---")
    print(f"Gap Distance (Absolute):    {dist_abs:.4e} m^-1")
    print(f"Gap Vector (Fractional):    [{gap_vec_frac[0]:.4f}, {gap_vec_frac[1]:.4f}, {gap_vec_frac[2]:.4f}]")
    print(f"Radii of Curvature:")
    print(f"  - Band {band1}: {rho1:.4e} m^-1")
    print(f"  - Band {band2}: {rho2:.4e} m^-1")
    print(f"Critical Field B0:          {b0_str}")
    
    if args.plot:
        print("\nGenerating 3D visualization...")
        plotter = pv.Plotter()
        
        bz_poly = pv.PolyData(bz_vertices).delaunay_3d().extract_surface()
        plotter.add_mesh(bz_poly, color="gray", opacity=0.1, label="First BZ")
        
        for edge in bz_ridges:
            plotter.add_mesh(pv.MultipleLines(points=edge), color="gray", line_width=1)
        
        plotter.add_mesh(pv.MultipleLines(points=loop1), color="red", line_width=5, label=f"Orbit {band1}")
        plotter.add_mesh(pv.MultipleLines(points=loop2), color="blue", line_width=5, label=f"Orbit {band2}")
        plotter.add_mesh(pv.Line(p1, p2), color="black", line_width=5, label="Gap")
        
        plotter.set_background("white")
        plotter.camera.azimuth = args.azimuth
        plotter.camera.elevation = args.elevation
        plotter.show()

def main():
    args = args_parser().parse_args()
    
    if args.mode == "skeaf":
        if args.bands is None:
            print("Error: SKEAF mode requires --bands argument (e.g., -b '[10, 11]')")
            return
        analyze_skeaf_mode(args)
    elif args.mode == "plane":
        analyze_plane_mode(args)

if __name__ == "__main__":
    main()
