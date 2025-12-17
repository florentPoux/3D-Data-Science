"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 06

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
Created on Tue April 15 14:15:09 2023

@author: Florent Poux
"""

#%% 1. Library setup
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#%% 2. Point Cloud Import

pcd = o3d.io.read_point_cloud("../DATA/techshop.ply")

#%% 3. Data Pre-Processing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)
o3d.visualization.draw_geometries([pcd])

# 3.1. Sampling
#%% 3.1. Random Sampling Test
retained_ratio = 0.01
sampled_pcd = pcd.random_down_sample(retained_ratio)
o3d.visualization.draw_geometries([sampled_pcd], window_name = "Random DownSampling")

#%% 3.2. Voxel-based Sampling Test

# Perform voxel grid downsampling
sub_voxel_pcd = pcd.voxel_down_sample(voxel_size=0.1)
o3d.visualization.draw_geometries([sub_voxel_pcd], window_name = "Voxel DownSampling") 


#%% 3.3 Spatial

from scipy.spatial import KDTree

def subsample_point_cloud_kdtree(input_cloud, radius=0.1, min_points=1):

    # Convert points to numpy array
    points = np.asarray(input_cloud.points)
    
    # Create KDTree
    kdtree = KDTree(points)
    
    # Mask to keep track of points to retain
    keep_mask = np.ones(len(points), dtype=bool)
    
    # Check neighborhood for each point
    for i, point in enumerate(points):
        # Find indices of points within radius
        neighbors = kdtree.query_ball_point(point, r=radius)
        
        # Remove point if not enough neighbors
        if len(neighbors) <= min_points:
            keep_mask[i] = False
    
    # Select points to keep
    subsampled_points = points[keep_mask]
    
    # Create new point cloud
    subsampled_cloud = o3d.geometry.PointCloud()
    subsampled_cloud.points = o3d.utility.Vector3dVector(subsampled_points)
    
    # Transfer colors if original cloud had colors
    if input_cloud.has_colors():
        colors = np.asarray(input_cloud.colors)
        subsampled_cloud.colors = o3d.utility.Vector3dVector(colors[keep_mask])
    
    # Transfer normals if original cloud had normals
    if input_cloud.has_normals():
        normals = np.asarray(input_cloud.normals)
        subsampled_cloud.normals = o3d.utility.Vector3dVector(normals[keep_mask])
    
    return subsampled_cloud

sub_spatial_pcd = subsample_point_cloud_kdtree(sub_voxel_pcd, radius=0.1, min_points=1)
o3d.visualization.draw_geometries([sub_spatial_pcd], window_name = "Spatial DownSampling")

#%% 3.2. Statistical outlier filter 
nn = 16
std_multiplier = 2

#The statistical outlier removal filter returns the point cloud and the point indexes
filtered_pcd, filtered_idx = sampled_pcd.remove_statistical_outlier(nn, std_multiplier)

#Visualizing the points filtered
outliers = sampled_pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([filtered_pcd, outliers])

#%% 3.3. Voxel downsampling
voxel_size = 0.05

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])

#%% 3.4. Estimating normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
print(nn_distance)
#setting the radius search to compute normals
radius_normals=nn_distance*4

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

# 3.3. Visualizing the point cloud in Python
pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled,outliers])