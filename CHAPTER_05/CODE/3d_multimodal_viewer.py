"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 05

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
* License: (c) learngeodata.eu
* Status: Hidden Perk

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Python 3.10 (Open3D Compatible)
* Libraries as described

Tested on Windows 11 and MacOS

🎵 Note: Have fun with this Code Solution.
Created on Mon Dec  9 17:56:31 2024
"""

#%% Importing libraries

import numpy as np
import pandas as pd
import open3d as o3d
from shapely.geometry import Polygon

print(f"Open 3D Version: {o3d.__version__}")

#%% Loading 3D datasets

data_folder="../DATA/"
pc_dataset="30HZ1_18_sampled.xyz"
mesh_dataset="NL.IMBAG.Pand.0637100000139735.obj"
result_folder="../DATA/RESULTS/"

"""We can prepare the point cloud by first creating a Pandas DataFrame object called pcd_df, which will host the point cloud data:"""

pcd_df= pd.read_csv(data_folder+pc_dataset, delimiter=";")
print(pcd_df.columns)

#%% Preparing our dataset with Panda

#1. Profiling to know the number, and description

pcd_df.shape
#returns (7103848, 7)
pcd_df.describe()
"""
                  X             Y  ...             B  Classification
count  7.103848e+06  7.103848e+06  ...  7.103848e+06    7.103848e+06
mean   9.251568e+04  4.518944e+05  ...  9.316822e+01    2.234453e+00
std    3.048018e+02  3.601502e+02  ...  2.958742e+01    2.907565e+00
min    9.198000e+04  4.512300e+05  ...  0.000000e+00    1.000000e+00
25%    9.224665e+04  4.516035e+05  ...  7.200000e+01    1.000000e+00
50%    9.254172e+04  4.518916e+05  ...  8.700000e+01    2.000000e+00
75%    9.278134e+04  4.522051e+05  ...  1.090000e+02    2.000000e+00
max    9.302000e+04  4.525200e+05  ...  2.550000e+02    2.600000e+01
"""
#We are going to decimate the point cloud

#%% 3.1. Sampling: decimation
pcd_subsampled = pcd_df.iloc[::2, :]

#%% Preparing our datasets
"""Numpy to Open3D"""
pcd_o3d=o3d.geometry.PointCloud()
pcd_o3d.points=o3d.utility.Vector3dVector(np.array(pcd_subsampled[['X','Y','Z']]))
pcd_o3d.colors=o3d.utility.Vector3dVector(np.array(pcd_subsampled[['R','G','B']])/255)

"""Loading the Mesh dataset"""
mesh=o3d.io.read_triangle_mesh(data_folder+mesh_dataset)
mesh.paint_uniform_color([0.1, 0.4, 0.8])

#Visualizing
o3d.visualization.draw_geometries([pcd_o3d,mesh])

#%% 3. [Optional Centering] Data Pre-Processing
pcd_center = pcd_o3d.get_center()
pcd_o3d.translate(-pcd_center)
mesh.translate(-pcd_center)

o3d.visualization.draw_geometries([pcd_o3d,mesh])

#%% Adding features (normals)

pcd_o3d.estimate_normals()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([pcd_o3d,mesh])

#%% 3.3. Visualizing with classification in Python

"""Coloring based on classification.
1=unclassified, 2=Ground, 6=building, 9=water, 26=rest
"""
pcd_subsampled['Classification'].unique()
colors=np.zeros((len(pcd_subsampled), 3))
colors[pcd_subsampled['Classification'] == 1] = [0.611, 0.8, 0.521]
colors[pcd_subsampled['Classification'] == 2] = [0.8, 0.670, 0.521]
colors[pcd_subsampled['Classification'] == 6] = [0.901, 0.419, 0.431]
colors[pcd_subsampled['Classification'] == 9] = [0.564, 0.850, 0.913]
colors[pcd_subsampled['Classification'] == 26] = [0.694, 0.662, 0.698]
pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_o3d,mesh])

#%% 4. 3D Python Challenges
#%% Challenge 1: POI Query

dist_POI=50
POI=mesh.get_center()

pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
[k, idx, _] = pcd_tree.search_radius_vector_3d(POI, dist_POI)
pcd_selection=pcd_o3d.select_by_index(idx)

o3d.visualization.draw_geometries([pcd_selection,mesh])

#%%"""## Challenge 2: Parcel Surface"""

o3d.visualization.draw_geometries_with_vertex_selection([pcd_selection])
#%%
o3d_parcel_corners=pcd_selection.select_by_index([10706 ,14956 ,15201 ,14043 ,1783 ,870])
o3d_parcel_corners=np.array(o3d_parcel_corners.points)[:,:2]
Polygon(o3d_parcel_corners)

pgon = Polygon(o3d_parcel_corners)
print(f"This is the obtained parcel area: {pgon.area} m²")

def sort_coordinates(XY):
    cx, cy = XY.mean(0)
    x, y = XY.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(-angles)
    return XY[indices]

np_sorted_2D_corners=sort_coordinates(o3d_parcel_corners)
pgon = Polygon(np_sorted_2D_corners)
Polygon(np_sorted_2D_corners)

print(f"This is the parcel area: {pgon.area} m²")

#%% Challenge 3: High and Low POI

print(pcd_selection.get_max_bound())
print(pcd_selection.get_min_bound())

np_pcd_selection=np.array(pcd_selection.points)
lowest_point_index=np.argmin(np_pcd_selection[:,2])
highest_point_index=np.argmax(np_pcd_selection[:,2])

low_point=pcd_selection.points[lowest_point_index]
high_point=pcd_selection.points[highest_point_index]

o3d.visualization.draw_geometries([pcd_selection])

lp=o3d.geometry.TriangleMesh.create_sphere()
hp=o3d.geometry.TriangleMesh.create_sphere()
lp.translate(np_pcd_selection[lowest_point_index])
hp.translate(np_pcd_selection[highest_point_index])

lp.compute_vertex_normals()
lp.paint_uniform_color([0.8,0.3,0.2])
hp.compute_vertex_normals()
hp.paint_uniform_color([0.1,0.3,0.8])
o3d.visualization.draw_geometries([pcd_selection,lp,hp])

#%% Point Cloud Voxelization
#1=unclassified, 2=Ground, 6=building, 9=water, 26=rest
df_vox = pcd_subsampled[pcd_subsampled['Classification'] == 2]
color_vox = np.tile([0.8, 0.670, 0.521], (len(df_vox), 1))

pcd_vox=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(df_vox[['X','Y','Z']])))
pcd_vox.colors=o3d.utility.Vector3dVector(color_vox)
pcd_vox.translate(-pcd_center)

o3d.visualization.draw_geometries([pcd_vox])

#%% Point Cloud Voxelization

pcd_tree_vox = o3d.geometry.KDTreeFlann(pcd_vox)
[k, idx_vox, _] = pcd_tree_vox.search_radius_vector_3d(POI, dist_POI)
pcd_selection_vox=pcd_vox.select_by_index(idx_vox)

o3d.visualization.draw_geometries([pcd_selection_vox,mesh])

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_selection_vox, voxel_size=2)
o3d.visualization.draw_geometries([voxel_grid])

o3d.visualization.draw_geometries([pcd_selection,lp,hp, mesh, voxel_grid])

#%%Adapting the coloring scheme: RGB
"""Adapting the colouring scheme: Black and Red, and the selection with another POI if needed."""

colors=np.zeros((len(pcd_subsampled), 3))
colors[pcd_subsampled['Classification'] == 6] = [1, 0, 0]
pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
dist_POI=150
POI=mesh.get_center()
pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)
[k, idx, _] = pcd_tree.search_radius_vector_3d(POI, dist_POI)
pcd_selection=pcd_o3d.select_by_index(idx)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_selection, voxel_size=2)
o3d.visualization.draw_geometries([voxel_grid])

idx_voxels=[v.grid_index for v in voxel_grid.get_voxels()]
color_voxels=[v.color for v in voxel_grid.get_voxels()]
bounds_voxels=[np.min(idx_voxels, axis=0),np.max(idx_voxels, axis=0)]
print(bounds_voxels)

#%% Challenge 4: Built Coverage

max_voxel={}
max_color={}

for idx, v in enumerate(idx_voxels):
    if (v[0],v[1]) in max_voxel.keys():
        if v[2]>max_voxel[(v[0],v[1])]:
            max_voxel[(v[0],v[1])]=v[2]
            max_color[(v[0],v[1])]=color_voxels[idx]
    else:
        max_voxel[(v[0],v[1])]=v[2]
        max_color[(v[0],v[1])]=color_voxels[idx]

count_building_coverage,count_non_building=0,0
for col in list(max_color.values()):
    if np.all(col==0):
        count_non_building+=1
    else:
        count_building_coverage+=1

print(f"Coverage of Buildings: {count_building_coverage*4} m²")
print(f"Coverage of the Rest: {count_non_building*4} m²")
print(f"Built Ratio: {(count_building_coverage*4)/(count_building_coverage*4+count_non_building*4)} m²")

#%% 5. Data Export"""

#Exporting the selection
o3d.io.write_point_cloud(result_folder+pc_dataset.split(".")[0]+"_result_filtered_o3d.ply", pcd_selection, write_ascii=False, compressed=False, print_progress=False)

#Exporting the Parcel Area
np.savetxt(result_folder+pc_dataset.split(".")[0]+"_selection.xyz", np.asarray(o3d_parcel_corners),delimiter=';', fmt='%1.9f')