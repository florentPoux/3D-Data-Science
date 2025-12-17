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

#%% 1. Initialization

# Libraries used
import numpy as np
import open3d as o3d
import laspy as lp

#%% 2. Dataset Preparation

# Gather the dataset
# Prepare the folder structure

# Create paths and Load Data
pcd_path ="../DATA/scanned_table.las"

point_cloud = lp.read(pcd_path)
xyz = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
rgb = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()/65535

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

#%% 3. Creating a Voxel Grid

vsize=max(pcd.get_max_bound()-pcd.get_min_bound())*0.005
vsize=round(vsize,4)

voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=vsize)
bounds=voxel_grid.get_max_bound()-voxel_grid.get_min_bound()

o3d.visualization.draw_geometries([voxel_grid])

#%% 4. Generating a single Voxel Entity

cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
cube.paint_uniform_color([1,0,0])
cube.compute_vertex_normals()
o3d.visualization.draw_geometries([cube])
                                          
#%% 5. Automate and Loop to create one Voxel Dataset

#grid index = integer value in a canonical space defined by the bounds
#use unit of 1 for voxels
voxels=voxel_grid.get_voxels()
vox_mesh=o3d.geometry.TriangleMesh()

for v in voxels:
    cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube.paint_uniform_color(v.color)
    cube.translate(v.grid_index, relative=False)
    vox_mesh += cube

o3d.visualization.draw_geometries([vox_mesh])

#%% 6. [OPTIONAL] Add Normals

vox_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([vox_mesh])

#%% 7. Voxel Dataset Post-Processing

#to align to the center of a cube of dimension 1
vox_mesh.translate([0.5,0.5,0.5], relative=True)

# to scale
vox_mesh.scale(vsize, [0,0,0])

# to translate
vox_mesh.translate(voxel_grid.origin, relative=True)

# To correct close vertices
vox_mesh.merge_close_vertices(0.0000001)

#%% 8. Voxel Dataset Exports (3D Mesh)

o3d.io.write_triangle_mesh("../RESULTS/voxel_mesh_heerlen_standard.ply", vox_mesh)

#◘Rotate and export
T = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0],[0, 0, 0, 1]])
o3d.io.write_triangle_mesh("../RESULTS/voxel_mesh_heerlen_rotated.ply", vox_mesh.transform(T))


#%% 9. Other approach

points = np.array(pcd.points)

#Voxel Size in meters
voxel_size= vsize
nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)

non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
idx_pts_vox_sorted=np.argsort(inverse)

voxel_grid={}
grid_barycenter,grid_candidate_center=[],[]
last_seen=0

for idx,vox in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)]= points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
    grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
    # grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] -np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
    last_seen+=nb_pts_per_voxel[idx]
    
np.savetxt("../RESULTS/voxel-best_point_%s.xyz" % (voxel_size), grid_barycenter, delimiter=";", fmt="%s")