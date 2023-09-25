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
def read_bxsf(file_name):
    #open and read file
    f = open(file_name, "r")
    lines=f.readlines()

    """
    this corresponds to code for gettinf the fermi eneryg
    """
    # get fermi energy
    x = lines[1].split(":     ")
    x = x[1].split("\n")
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
    eigen_text = lines_conc.split(' BAND:   ' + band_index)[1]
    eigen_text = eigen_text.split('END_BANDGRID_3D')[0]
    eigen_vals = eigen_text.split(" ")
    eigen_vals = [val.strip("\n") for val in eigen_vals]
    eigen_vals = eigen_vals[1:-1]
    eigen_vals = [float(val) for val in eigen_vals]
    
    eigenvalues = np.array(eigen_vals).reshape(dimensions)

    eig_vals = np.zeros(dimensions)
    for k in range(dimensions[0]):
        for j in range(dimensions[1]):
            for i in range(dimensions[2]):
                eig_vals[i,j,k] = eigen_vals[dimensions[0]*dimensions[1]*i + dimensions[1]*j + k]


    
    """
    Get basis vectors
    """
    #first spanning vector
    vec_1 = lines[9].split("      ")
    vec_1 = [val.strip("\n") for val in vec_1]
    vec_1 = vec_1[1:]
    #convert dimesions to integer values
    vec_1 = [float(val) for val in vec_1]

    #second spanning vector
    vec_2 = lines[10].split("      ")
    vec_2 = [val.strip("\n") for val in vec_2]
    vec_2 = vec_2[1:]
    #convert dimesions to integer values
    vec_2 = [float(val) for val in vec_2]

    #thrid spanning vector
    vec_3 = lines[11].split("      ")
    vec_3 = [val.strip("\n") for val in vec_3]
    vec_3 = vec_3[1:]
    #convert dimesions to integer values
    vec_3 = [float(val) for val in vec_3]

    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    vec_3 = np.array(vec_3)

    x_vals = []
    y_vals = []
    z_vals = []
    
    k_vectors = np.zeros(dimensions + [3])
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                k_vectors[i,j,k,:] = i*vec_1 + j*vec_2 + k*vec_3

                x_vals += [k_vectors[i,j,k,0]]
                y_vals += [k_vectors[i,j,k,1]]
                z_vals += [k_vectors[i,j,k,2]]



    grid = pv.StructuredGrid(np.array(x_vals), np.array(y_vals), np.array(z_vals))
    grid.dimensions = [dimensions[0], dimensions[1], dimensions[2]]
    # generate data grid for computing the values
    #X, Y, Z = np.mgrid[-5:5:18j, -5:5:18j, -5:5:18j]
    #values = X**2 * 0.5 + Y**2 + Z**2 * 2

    # create a structured grid
    # (for this simple example we could've used an unstructured grid too)
    # note the fortran-order call to ravel()!
    #grid = pv.StructuredGrid(X, Y, Z)
    grid.point_arrays['values'] = eig_vals.flatten()  # also the active scalars


    # compute 2 isosurfaces
    isos = grid.contour(isosurfaces=2, rng=[e_f,e_f])
    # or: mesh.contour(isosurfaces=np.linspace(10, 40, 3)) etc.
    
    cell = np.array([vec_1*dimensions[0], vec_2*dimensions[1], vec_3*dimensions[2]])
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
#get base file name
file_name = sys.argv[1]

files = glob.glob(file_name + "*bxsf.band-*")

k_vectors, eig_vals, e_f, cell, dimensions, isos = read_bxsf(files[0])


#generate BZ from voronoi analysis
v, e, f = get_brillouin_zone_3d(cell)

#triangulate BZ surface
bz_surf = pv.PolyData(v)
bz_surf = bz_surf.delaunay_3d()
bz_surf = bz_surf.extract_surface()
edges = bz_surf.extract_all_edges()


color_list = sns.color_palette("hls", len(files))
color_list = ["#73D2DE", "#f5b105"]
counter = 0
for file in files:
        
        k_vectors, eig_vals, e_f, cell, dimensions, isos = read_bxsf(file)
        vec1 = cell[0]*(dimensions[0]-1)/dimensions[0]
        vec2 = cell[1]*(dimensions[1]-1)/dimensions[1]
        vec3 = cell[2]*(dimensions[2]-1)/dimensions[2]

        
        
        try:
            isos = (isos 
                    + isos.translate(-vec1, inplace=False) 
                    + isos.translate(-vec2, inplace=False) 
                    + isos.translate(-vec3, inplace=False) 
                    + isos.translate(-vec1-vec2, inplace=False) 
                    + isos.translate(-vec1-vec3, inplace=False)
                    + isos.translate(-vec2-vec3, inplace=False)
                    + isos.translate(-vec1-vec2-vec3, inplace=False)
            )
            isos = isos.clip_surface(bz_surf, invert=True)
            plotter.add_mesh(isos, lighting=True, color = color_list[counter], opacity=1.0)
        except:
            #print(file + " contains an empty mesh")
            pass
        counter += 1


#plot BZ
for xx in e:
    line = pv.MultipleLines(points = np.array([xx[:, 0], xx[:, 1], xx[:, 2]]).T)
    plotter.add_mesh(line, color = "black", line_width = 2) 
    
plotter.set_background('white')
plotter.set_position([0, 0, 13])
plotter.save_graphic("FS_top.pdf")

plotter.set_background('white')
plotter.camera_position = 'yz'
plotter.set_position([13, 0, 0])
plotter.camera.azimuth = 65
plotter.camera.elevation = 30
plotter.save_graphic("FS_side.pdf")


plotter.show()

