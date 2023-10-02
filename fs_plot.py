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
import matplotlib
from scipy.linalg import norm
import pyvista as pv
import pandas as pd
import seaborn as sns
import copy
import re
import glob
import sys
import scienceplots
from scipy.ndimage import map_coordinates
import argparse
import ast

#define some plotting parameters for rough
#plotting of output 
plt.style.use(['nature'])
cp = sns.color_palette("Set2")
#pv.rcParams['transparent_background'] = True


"""
Reads .bxsf file and determines two
matrices, one corresponding to the eigenvalues
and the other the k space vectors

returns:
    k_vectors
    eigenvalues
    ef
"""
def read_bxsf(file_name, scale, delta_energy):
    #open and read file
    f = open(file_name, "r")
    lines=f.readlines()

    """
    this corresponds to code for gettinf the fermi eneryg
    """
    # get fermi energy
    x = lines[1].split(" ")
    x = [val for _,val in enumerate(x) if val!='']
    x = [val for _,val in enumerate(x) if val!='\t']
    x = x[-1].split("\n")
    # convert ef to a number
    e_f = float(x[0])

    # get fermi energy
    band_index = re.findall(r'\d+', lines[12])[0]

    """
    Now get the dimensions of the k mesh
    """

    # get dimensions
    x = re.findall(r'\d+', lines[7])
    #convert dimesions to integer values
    dimensions = [int(dim) for dim in x]

    """
    Now extract eigenvalues
    """

    #get eerything after band index
    lines_conc = ''.join(lines)
    for line in lines:
        if 'BAND' in line and band_index in line:
            eigen_text = lines_conc.split(line)[1]
            break
    eigen_text = eigen_text.split('END_BANDGRID_3D')[0]
    eigen_vals = eigen_text.split(" ")
    eigen_vals = [val.strip("\n") for val in eigen_vals]
    eigen_vals = eigen_vals[1:-1]
    eigen_vals = [float(val) for val in eigen_vals]
    
    eigenvalues = np.array(eigen_vals).reshape(dimensions)

    eig_vals = np.zeros(dimensions)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                eig_vals[i,j,k] = eigen_vals[dimensions[2]*dimensions[1]*i + dimensions[2]*j + k]
  
    """
    Get basis vectors
    """
    #first spanning vector
    vec_1 = lines[9].split(" ")
    vec_1 = [val.strip("\n") for val in vec_1]
    vec_1 = [val.strip("\t") for val in vec_1]
    vec_1 = [val for _,val in enumerate(vec_1) if val!='']
    #convert dimesions to integer values
    vec_1 = [float(val) for val in vec_1]

    #second spanning vector
    vec_2 = lines[10].split(" ")
    vec_2 = [val.strip("\n") for val in vec_2]
    vec_2 = [val.strip("\t") for val in vec_2]
    vec_2 = [val for _,val in enumerate(vec_2) if val!='']
    #convert dimesions to integer values
    vec_2 = [float(val) for val in vec_2]

    #thrid spanning vector
    vec_3 = lines[11].split(" ")
    vec_3 = [val.strip("\n") for val in vec_3]
    vec_3 = [val.strip("\t") for val in vec_3]
    vec_3 = [val for _,val in enumerate(vec_3) if val!='']
    #convert dimesions to integer values
    vec_3 = [float(val) for val in vec_3]

    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    vec_3 = np.array(vec_3)


    if scale != 1:

        #interpolation here
        dimensions_int = [int(scale*dimension) for dimension in dimensions]
        x_vals_int = np.linspace(0,dimensions_int[0] - 1, dimensions_int[0])
        y_vals_int = np.linspace(0,dimensions_int[1] - 1, dimensions_int[1])
        z_vals_int = np.linspace(0,dimensions_int[2] - 1, dimensions_int[2])

        dx = x_vals_int[1]-x_vals_int[0]
        dy = y_vals_int[1]-y_vals_int[0]
        dz = z_vals_int[1]-z_vals_int[0]

        scaling = np.array([dx, dy, dz])

        x_vals_int, y_vals_int, z_vals_int = np.meshgrid(x_vals_int, y_vals_int, z_vals_int)
        out_coords = np.array((x_vals_int.ravel(), y_vals_int.ravel(), z_vals_int.ravel()))

        #x_vals_int, y_vals_int, z_vals_int = np.meshgrid(x_vals_int, y_vals_int, z_vals_int)

        out_data = map_coordinates(eig_vals, out_coords/scale, order=3, mode = 'reflect')
        
        out_data = out_data.reshape(dimensions_int[0], dimensions_int[1], dimensions_int[2])

        out_grid_x, out_grid_y, out_grid_z = np.meshgrid(range(-dimensions_int[0], dimensions_int[0]), range(-dimensions_int[1], dimensions_int[1]), range(-dimensions_int[2], dimensions_int[2]))

    else:

        out_data = eig_vals
        dimensions_int = dimensions
        out_grid_y, out_grid_x, out_grid_z = np.meshgrid(range(-dimensions_int[0], dimensions_int[0]), range(-dimensions_int[1], dimensions_int[1]), range(-dimensions_int[2], dimensions_int[2]))

    out_data = np.tile(out_data, (2,2,2))

    x_vals = []
    y_vals = []
    z_vals = []
    
    k_vectors = np.zeros([2*dimension for dimension in dimensions_int] + [3])

    k_vectors[:] = (np.outer(out_grid_x+scale/2, vec_1).reshape([2*dimension for dimension in dimensions_int] + [3]) + np.outer(out_grid_y+scale/2,vec_2).reshape([2*dimension for dimension in dimensions_int] + [3]) + np.outer(out_grid_z+scale/2, vec_3).reshape([2*dimension for dimension in dimensions_int] + [3]))/scale

    k_vectors = k_vectors.reshape(8*dimensions_int[0]*dimensions_int[1]*dimensions_int[2],3)
    x_vals = k_vectors[:,0]
    y_vals = k_vectors[:,1]
    z_vals = k_vectors[:,2]

    grid = pv.StructuredGrid(np.array(x_vals)/dimensions[0], np.array(y_vals)/dimensions[1], np.array(z_vals)/dimensions[2])
    grid.dimensions = [2*dimensions_int[2], 2*dimensions_int[1], 2*dimensions_int[0]]
    # generate data grid for computing the values
    #X, Y, Z = np.mgrid[-5:5:18j, -5:5:18j, -5:5:18j]
    #values = X**2 * 0.5 + Y**2 + Z**2 * 2

    # create a structured grid
    # (for this simple example we could've used an unstructured grid too)
    # note the fortran-order call to ravel()!
    #grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data['values'] = out_data.flatten()  # also the active scalars


    # compute 2 isosurfaces
    iso1 = grid.contour(isosurfaces=1, rng=[e_f-delta_energy,e_f-delta_energy])
    iso2 = grid.contour(isosurfaces=1, rng=[e_f+delta_energy,e_f+delta_energy])
    isos = [iso1, iso2]
    # or: mesh.contour(isosurfaces=np.linspace(10, 40, 3)) etc.

    dimensions = dimensions_int
    
    cell = np.array([vec_1, vec_2, vec_3])
    return k_vectors, eig_vals, e_f, cell, dimensions, isos



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

        if(pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))
    
    return vor.vertices[bz_vertices], bz_ridges, bz_facets


plotter = pv.Plotter()


#parse command line arguments
parser = argparse.ArgumentParser(description="Fermi Surface Plotter",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--name", metavar='\b', type=str, help="Name of the bxsf file in which data is stored", required=True)
parser.add_argument("-i", "--interpolation-factor", metavar='\b', type=float, default=1, help="Degree of interpolation of Fermi surface")
parser.add_argument("-b", "--bands", metavar='\b', type=str, help="List of bands to include in plot")
parser.add_argument("-z", "--zoom", metavar='\b', type=float, help="Zoom factor of saved image", default=1.0)
parser.add_argument("-de", "--delta-energy", metavar='\b', type=float, help="Fermi energy shift for plotting different colors for inner and outer sheets", default=0.00005)


args = parser.parse_args()

#assign parsed command line arguments
file_name = args.name
scale = args.interpolation_factor


#load up bxsf files
if args.bands is not None:
    bands = ast.literal_eval(args.bands)
    files = []
    for band in bands:
        try:
            files += glob.glob(file_name + "*bxsf.band-" + str(band))
        except:
            print(f"Error: No matching band indices found for band {band}, check band indices")
            exit()
else:
    files = glob.glob(file_name + "*bxsf.band-*")

if len(files) == 0:
    print("Error: No .bxsf files found, check file name")
    exit()

if scale < 0.5:
    print("Error: Please choose a scaling factor greater than 0.5")
    exit()

k_vectors, eig_vals, e_f, cell, dimensions, isos = read_bxsf(files[0], scale=scale, delta_energy=args.delta_energy)


#generate BZ from voronoi analysis
v, e, f = get_brillouin_zone_3d(cell)

#triangulate BZ surface
bz_surf = pv.PolyData(v)
bz_surf = bz_surf.delaunay_3d()
bz_surf = bz_surf.extract_surface()
edges = bz_surf.extract_all_edges()


color_list = sns.color_palette("hls", 2*len(files))
counter = 0
for file in files:
        
        print(file)
        
        k_vectors, eig_vals, e_f, cell, dimensions, isos = read_bxsf(file, scale, delta_energy=args.delta_energy)
        vec1 = cell[0]*(dimensions[0]-scale)/dimensions[0]
        vec2 = cell[1]*(dimensions[1]-scale)/dimensions[1]
        vec3 = cell[2]*(dimensions[2]-scale)/dimensions[2]

        plotter_ind = pv.Plotter()

        for iso in isos:
            
            try:

                iso = iso.clip_surface(bz_surf, invert=True)

                plotter.add_mesh(iso, lighting=True, color = color_list[counter], opacity=1.0, clim = [-1, 2])
                plotter_ind.add_mesh(iso, lighting=True, color = color_list[counter], opacity=1.0, clim = [-1, 2])
            except:
                #print(file + " contains an empty mesh")
                pass

            for xx in e:
                line = pv.MultipleLines(points = np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T)
                plotter_ind.add_mesh(line, color = "black", line_width = 2) 

            plotter_ind.set_background('white')
            plotter_ind.camera_position = 'yz'
            plotter_ind.set_position([0.5/args.zoom, 0, 0])
            plotter_ind.camera.azimuth = 65
            plotter_ind.camera.elevation = 30
            plotter_ind.save_graphic("FS_side_" + file[-7:] + ".pdf")

            counter += 1



#plot BZ
for xx in e:
    line = pv.MultipleLines(points = np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T)
    plotter.add_mesh(line, color = "black", line_width = 2) 
    
plotter.set_background('white')
plotter.set_position([0, 0, 0.5/args.zoom])
plotter.save_graphic("FS_top.pdf")

plotter.set_background('white')
plotter.camera_position = 'yz'
plotter.set_position([0.5/args.zoom, 0, 0])
plotter.camera.azimuth = 65
plotter.camera.elevation = 30
plotter.save_graphic("FS_side.pdf")


plotter.show()

