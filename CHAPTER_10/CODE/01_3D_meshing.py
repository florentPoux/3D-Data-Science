"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 10

General Information:
-------------------
* 🦊 Created by:    Florent Poux
* 📅 Last Update:   Dec. 2025
* © Copyright:      Florent Poux
* 📜 License:       MIT

Dependencies:
------------
* Environment:      Anaconda or Miniconda
* Python Version:   3.9+
* Key Libraries:    NumPy, Pandas, Open3D, Laspy, Scikit-Learn

Helpful Links:
-------------
* 🏠 Author Website:        https://learngeodata.eu
* 📚 O'Reilly Book Page:    https://www.oreilly.com/library/view/3d-data-science/9781098161323/

Enjoy this code! 🚀
"""
# -*- coding: utf-8 -*-
"""
General Information
* Created by: 🦊 Florent Poux. 
* Copyright: Florent Poux.
* License: MIT
* Status: Online

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Have fun with this Code Solution.

🎵 Note: Styling was not taken care of at this stage.

Enjoy!
"""

#%% Step 1: Setting up the environment

import numpy as np
import open3d as o3d

#%% Step 2: Load and prepare the data"""

#create paths and load data
input_path="../DATA/"
output_path="../RESULTS/"
dataname="sample_w_normals.xyz"
point_cloud= np.loadtxt(input_path+dataname,skiprows=1)

#Format to open3d usable objects
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])

o3d.visualization.draw_geometries([pcd],window_name="Point Cloud")

#%% Step 3: Choose a meshing strategy

"Now we are ready to start the surface reconstruction process by meshing the pcd point cloud. I will give my favorite way to efficiently obtain results, but before we dive in, some condensed details ar necessary to grasp the underlying processes. I will limit myself to two meshing strategies. See article"

#%% Step 4: Process the data
## Strategy 1: BPA

#Radius determination
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

#Computing the meshs
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

o3d.visualization.draw_geometries([bpa_mesh],window_name="BPA Mesh")

#decimating the mesh
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

"""*Optional ---*"""

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

o3d.visualization.draw_geometries([dec_mesh],window_name="BPA Mesh Post-Processed")

#%%# Strategy 2: Poisson' reconstruction"""

#computing the mesh
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

o3d.visualization.draw_geometries([poisson_mesh],window_name=" Mesh Poisson")

#cropping
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)

o3d.visualization.draw_geometries([p_mesh_crop],window_name=" Mesh Poisson Cropped")

#%% Step 5: Visualization

p_mesh_crop.compute_triangle_normals()
o3d.visualization.draw_geometries([p_mesh_crop],window_name=" Mesh Poisson Cropped Normals")

#%% Step 6: Export
o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", dec_mesh)
o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)

#%%function creation
def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

#execution of function
my_lods = lod_mesh_export(bpa_mesh, [100000,50000,10000,1000,100], ".ply", output_path)

#execution of function
my_lods2 = lod_mesh_export(bpa_mesh, [8000,800,300], ".ply", output_path)