import glob
import ast
from scipy.spatial import Voronoi
import numpy as np
from scipy.ndimage import map_coordinates
import re
import pyvista as pv
import argparse

def load_files(args: argparse.Namespace) -> list[str]:
    """
    Function to load in bxsf files taking arguments to determine whether we have a spin polarised case

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        list[str]: List of files to be included in plots
    """

    # load up bxsf files
    if args.bands is not None:
        bands = ast.literal_eval(args.bands)
        files = []
        for band in bands:
            try:
                if args.spin == "u":
                    files += glob.glob(
                        args.name + "*bxsf.band-" + str(band) + "_up"
                    )
                elif args.spin == "d":
                    files += glob.glob(
                        args.name + "*bxsf.band-" + str(band) + "_up"
                    )
                elif args.spin == "ud":
                    files += glob.glob(
                        args.name + "*bxsf.band-" + str(band) + "_*"
                    )
                elif args.spin == "n":
                    files += glob.glob(args.name + "*bxsf.band-" + str(band))
            except:
                print(
                    f"Error: No matching band indices found for band {band}, check band indices"
                )
                exit()
    else:
        files = glob.glob(args.name + "*bxsf.band-*")
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


def read_bxsf_info(file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str, float, np.ndarray]:
    """
    Reads the bxsf file and extracts information such as Fermi energy, dimensions, and basis vectors

    Args:
        file_name (str): Path to the bxsf file

    Returns:
        tuple: Contains basis vectors, dimensions, band index, Fermi energy, and cell matrix
    """

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


def generate_energy_grid(file_name: str, scale: int, order: int, lines) -> tuple[pv.StructuredGrid, float, list[int], np.ndarray, np.ndarray]:
    """
    Reads the bxsf file and generates a structured grid for the energy values
    This function is used to create a grid for the energy values in the bxsf file.
    It extracts the basis vectors, dimensions, and Fermi energy from the file,
    and then creates a structured grid with the energy values interpolated based on the scale and order
    of interpolation.

    Args:
        file_name (str): Path to the bxsf file
        scale (int): Scaling factor for interpolation
        order (int): Order of interpolation
        lines (list[str]): Lines from the bxsf file

    Returns:
        tuple: Contains the structured grid, Fermi energy, dimensions, k vectors, and eigenvalues
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

    eig_vals = np.reshape(eigen_vals, dimensions, order="C")

    """    eig_vals = np.zeros(dimensions)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                eig_vals[i, j, k] = eigen_vals[
                    dimensions[2] * dimensions[1] * i + dimensions[2] * j + k
                ]"""

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
            x_vals_int, y_vals_int, z_vals_int, indexing="ij"
        )

        out_grid_z, out_grid_y, out_grid_x = np.meshgrid(
            range(-dimensions_int[2] + scale, dimensions_int[2] - scale + 1),
            range(-dimensions_int[1] + scale, dimensions_int[1] - scale + 1),
            range(-dimensions_int[0] + scale, dimensions_int[0] - scale + 1),
            indexing="ij",
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

        out_data = out_data.swapaxes(2, 0)

    else:

        out_data = eig_vals.swapaxes(2, 0)
        dimensions_int = dimensions
        out_grid_z, out_grid_y, out_grid_x = np.meshgrid(
            range(-dimensions_int[2] + 1, dimensions_int[2]),
            range(-dimensions_int[1] + 1, dimensions_int[1]),
            range(-dimensions_int[0] + 1, dimensions_int[0]),
            indexing="ij",
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
        2 * dimensions_int[0] + 1 - 2 * scale,
        2 * dimensions_int[1] + 1 - 2 * scale,
        2 * dimensions_int[2] + 1 - 2 * scale,
    ]

    # create a structured grid
    # (for this simple example we could've used an unstructured grid too)
    # note the fortran-order call to ravel()!
    # grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["values"] = out_data.flatten()  # also the active scalar

    return grid, e_f, dimensions_int, k_vectors, eig_vals, cell


def read_bxsf(file_name: str, scale: int, order: int, shift_energy: float, fermi_velocity: bool, scalar: str = "None") -> tuple[np.ndarray, np.ndarray, float, np.ndarray, list[int], list[pv.PolyData], pv.StructuredGrid]:
    """
    Reads .bxsf file and determines two matrices, one corresponding to the eigenvalues and the other the k space vectors

    Args:
        file_name (str): Path to the bxsf file
        scale (int): Scaling factor for interpolation
        order (int): Order of interpolation
        shift_energy (float): Shift in the Fermi energy
        fermi_velocity (bool): Whether to compute the Fermi velocity on the surface
        scalar (str, optional): Name of file containing a scalar field to plot on the Fermi surface. Defaults to "None".

    Returns:
        tuple: Contains k_vectors, eigenvalues, Fermi energy, cell matrix, dimensions, isosurfaces, and grid
    """
    # open and read file
    f = open(file_name, "r")
    lines = f.readlines()

    grid, e_f, dimensions_int, k_vectors, eig_vals, cell = generate_energy_grid(file_name, scale, order, lines)

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

def calculate_shift(file_names: list, scale: int, order: int, shift_energy: float, bz_surf: pv.StructuredGrid) -> tuple[np.float64, np.float64]:
    """
    Compute the energy shift for the second band so that the sum of enclosed
    volumes from two isosurfaces remains constant after shifting the first band.

    Args:
        file_names (list): [path_band1_bxsf, path_band2_bxsf]
        scale (int): Scaling factor for interpolation
        order (int): Order of interpolation
        shift_energy (float): Shift applied to band 1's Fermi level (in same units as e_f)
        bz_surf (pv.StructuredGrid): Brillouin zone surface for clipping

    Returns:
        tuple[np.float64, np.float64]:
            (delta2, carrier_vol)
            where delta2 is the required shift for band 2, and carrier_vol
            is the original total enclosed volume (vol1_orig + vol2_orig).
    """
    # ---------- Band 1: original & shifted volumes ----------
    # open and read file
    file_name = file_names[0]
    with open(file_name, "r") as f:
        lines1 = f.readlines()

    grid1, e_f1, dimensions_int1, k_vectors1, eig_vals1, cell1 = generate_energy_grid(file_name, scale, order, lines1)

    vol1_orig  = make_isovolume(grid1, e_f1,               scalars="values").clip_surface(bz_surf, invert=True).volume
    vol1_shift = make_isovolume(grid1, e_f1 + shift_energy, scalars="values").clip_surface(bz_surf, invert=True).volume

    # ---------- Band 2: original volume (grid reused for solver) ----------
    file_name = file_names[1]
    with open(file_name, "r") as f:
        lines2 = f.readlines()

    grid2, e_f2, dimensions_int2, k_vectors2, eig_vals2, cell2 = generate_energy_grid(file_name, scale, order, lines2)

    def vol2_at(delta2: float) -> float:
        return make_isovolume(grid2, e_f2 + delta2, scalars="values").clip_surface(bz_surf, invert=True).volume

    vol2_orig = vol2_at(0.0)

    # ---------- Target total & root function ----------
    carrier_vol = vol1_orig + vol2_orig
    target_v2   = carrier_vol - vol1_shift  # we want vol2(e_f2 + delta2) == target_v2

    def g(delta2: float) -> float:
        return vol2_at(delta2) - target_v2

    # ---------- Warm-start via local linearization (small central difference) ----------
    # This is just to pick a good initial bracket; final answer uses robust bisection.
    def dV2dE(h: float = 1e-3) -> float:
        vm = vol2_at(-h)
        vp = vol2_at(+h)
        return (vp - vm) / (2.0 * h)

    try:
        dV = dV2dE()
        # If derivative is tiny, fall back to a neutral guess
        delta2_guess = 0.0 if abs(dV) < 1e-16 else (target_v2 - vol2_orig) / dV
    except Exception:
        delta2_guess = 0.0

    # ---------- Bracket expansion around the guess ----------
    a = delta2_guess - 0.01
    b = delta2_guess + 0.01
    fa = g(a)
    fb = g(b)

    expand_count = 0
    max_expand = 14  # expands bracket size up to ~16384x if needed
    while fa * fb > 0 and expand_count < max_expand:
        # Exponentially widen the bracket, centered on the guess
        width = (b - a) * 2.0
        a = delta2_guess - width
        b = delta2_guess + width
        fa = g(a)
        fb = g(b)
        expand_count += 1

    if fa * fb > 0:
        raise RuntimeError(
            "Could not bracket a solution for delta2. "
            "Try widening the energy range or inspecting V2(E) monotonicity."
        )

    # ---------- Bisection solve ----------
    tol = 1e-8  # energy tolerance (adjust to your preferred precision)
    max_it = 200
    for _ in range(max_it):
        m = 0.5 * (a + b)
        fm = g(m)
        if abs(fm) < 1e-10 or 0.5 * (b - a) < tol:
            delta2 = m
            break
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    else:
        delta2 = 0.5 * (a + b)

    vol2_shift = make_isovolume(grid2, e_f2 + delta2, scalars="values").clip_surface(bz_surf, invert=True).volume

    return shift_energy, delta2


def get_brillouin_zone_3d(cell: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Uses the k-space vectors and Voronoi analysis to define the BZ of the system

    Args:
        cell (np.ndarray): A 3x3 matrix defining the basis vectors in reciprocal space

    Returns:
        tuple: Contains vertices of BZ, edges of the BZ, and BZ facets
    """

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

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

def make_isovolume(vol, vmax, vmin=-1e40, scalars=None):
    """
    vol: UniformGrid/ImageData/RectilinearGrid/UnstructuredGrid with a scalar field
    vmin,vmax: inclusive bounds for the isovolume
    scalars: name of the scalar array (defaults to active)
    prefer: 'point' or 'cells' — which data association to use for thresholding
    """
    # Ensure the intended scalars are active
    if scalars is not None:
        vol = vol.copy()
        vol.set_active_scalars(scalars)

    # Threshold returns an UnstructuredGrid (volumetric cells)
    iso_vol = vol.clip_scalar(
        scalars=scalars, value=vmin, invert=False,
    ).clip_scalar(
        scalars=scalars, value=vmax, invert=True,
    )

    return iso_vol