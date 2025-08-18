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
import glob
import scienceplots
from src.parser import *
from src.plotter import *
from src.skeaf_handler import *
from src.utils import *

pv.global_theme.colorbar_orientation = "vertical"

# define some plotting parameters for rough
# plotting of output
plt.style.use(["nature"])
cp = sns.color_palette("Set2")
# pv.rcParams['transparent_background'] = True


def main():
    # parse command line arguments
    parser = args_parser()
    args = parser.parse_args()

    # assign parsed command line arguments
    file_name = args.name
    scale = args.interpolation_factor
    order = args.order
    opacity = args.opacity
    formt = args.format
    shift_energy = args.shift_energy

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
        print("Error: Please choose a interpolation order between 1 and 5")
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
    if args.shift_energy_pair != 0.0 and len(files) != 2:
        print(
            "Error: Cannot use the shift energy pair option with more than two bands. Please use the shift energy option instead."
        )
        exit()
    if args.shift_energy_pair != 0.0 and args.shift_energy != 0.0:
        print(
            "Error: Cannot use both the shift energy and the shift energy pair option. Please choose one."
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
    
    if args.shift_energy_pair != 0.0:
        shift_energy_list = calculate_shift(files, scale, order, args.shift_energy_pair, bz_surf)
        print(f"Shift energy for first band: {shift_energy_list[0]}")
        print(f"Shift energy for second band: {shift_energy_list[1]}")


    for file in files:

        print(file)
        if shift_energy_list is not None:
            shift_energy = shift_energy_list[counter]
        else:
            shift_energy = args.shift_energy

        k_vectors, eig_vals, e_f, cell, dimensions, isos, _ = read_bxsf(
            file,
            scale,
            order=order,
            shift_energy=shift_energy,
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
            run_skeaf(file, file.split(".")[-1], args)
            orbits_list = plot_skeaf(file.split(".")[-1], args, cell)

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
                            backface_params={
                                "color": color_list[2 * counter + 1]
                            },
                        )

            except Exception as err: 
                print(err)
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
        line = pv.MultipleLines(
            points=np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T
        )
        plotter.add_mesh(
            line, color="black", line_width=args.resolution * args.line_width
        )
        if args.interactive == True:
            plotter_int.add_mesh(
                line, color="black", line_width=args.line_width
            )

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
        print(f"\tZo0.00023908157113949758om: {zoom}")


if __name__ == "__main__":
    main()
