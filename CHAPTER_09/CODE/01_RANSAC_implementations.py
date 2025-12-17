"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 09

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
# O'Reilly: 3D Data Science with Python
## Chapter 9 - 3D Shape Detection

General Information
* Created by: 🦊 Florent Poux. 
* Copyright: Florent Poux.
* License: MIT
* Status: Review Only (Confidential)

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Have fun with this Code Solution.

🎵 Note: Styling was not taken care of at this stage.

Enjoy!
"""

#%% 1. Libraries import

import numpy as np
import open3d as o3d

from sklearn.neighbors import KDTree

#%% Loading a dataset

pcd = o3d.io.read_point_cloud("../DATA/the_researcher_desk.ply")
points = np.asarray(pcd.points)

o3d.visualization.draw_geometries([pcd])

#%% Parameter estimation

tree = KDTree(points, leaf_size=2)

nearest_dist, nearest_ind = tree.query(points, k=8)

nearest_dist_mean = np.mean(nearest_dist[:,1:],axis=0)#if we want to average per nearest neighbor, for the k=8

nearest_dist_m = np.mean(nearest_dist[:,1:])

#%% Model fitting

sample = points[np.random.choice(points.shape[0], 3, replace=False)]

v1 = sample[1] - sample[0]
v2 = sample[2] - sample[0]
normal = np.cross(v1, v2)

a,b,c = normal / np.linalg.norm(normal)
d = -np.dot(normal, sample[1])

distances = np.dot(points, [a, b, c]) + d / np.sqrt(a**2 + b**2 + c**2)

threshold = nearest_dist_m
inliers = np.where(np.abs(distances) < threshold)[0]

#%% Making it a function

def ransac_plane(points, num_iterations=1000, threshold=0.1):
    best_inliers = []
    best_plane = None
    
    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        
        # Calculate plane equation ax + by + cz + d = 0
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        a, b, c = normal / np.linalg.norm(normal)
        # d = -np.dot(normal, sample[1])
        d = -np.sum(normal*sample[1])
        
        # Calculate distances of all points to the plane
        distances = np.dot(points, [a, b, c]) + d / np.sqrt(a**2 + b**2 + c**2)
        
        # Count inliers
        inliers = np.where(np.abs(distances) < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)
    
    return best_plane, best_inliers

#%% Executing RANSAC Planar detection

import time

t0 = time.time()
plane_params, plane_inliers = ransac_plane(points, 1000, nearest_dist_m )
t1 = time.time()

print(t1-t0)
print(f"Plane equation: {plane_params[0]}x + {plane_params[1]}y + {plane_params[2]}z + {plane_params[3]} = 0")

#%% Application 1: RANSAC for Segmentation

mask = np.ones(len(points), dtype=bool)
mask[plane_inliers] = False
outliers = points[mask]

plane_cloud = o3d.geometry.PointCloud()
plane_cloud.points = o3d.utility.Vector3dVector(points[plane_inliers])
plane_cloud.paint_uniform_color([0.8, 0.2, 0.6])

plane_outliers = o3d.geometry.PointCloud()
plane_outliers.points = o3d.utility.Vector3dVector(outliers)
plane_outliers.paint_uniform_color([0.3, 0.1, 0.9])

o3d.visualization.draw_geometries([plane_cloud, plane_outliers])




#%% Application 2: RANSAC for Analytics

datasets = ["../DATA/the_researcher_desk.ply", "../DATA/the_playground.ply"]

# Load point cloud (replace with your own point cloud file)
pcd = o3d.io.read_point_cloud(datasets[-1])
points = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])
#Compute threshold
tree = KDTree(points, leaf_size=2)
nearest_dist, nearest_ind = tree.query(points, k=8)
nearest_dist_mean = np.mean(nearest_dist[:,1:],axis=0)
threshold = np.max(nearest_dist_mean) + np.std(nearest_dist_mean)

#RANSAC Shape Detection
t0 = time.time()
plane_params, plane_inliers = ransac_plane(points, 1000, threshold)
t1 = time.time()

def angle_between_vectors(v1, v2):
    # Convert lists to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute the dot product
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_angle = dot_product / (v1_mag * v2_mag)
    
    # Use arccos to get the angle in radians
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return min(angle_deg, 180 - angle_deg)

angle = angle_between_vectors(plane_params[0:3], [0,0,1])
print(f"The angle of the plane is {angle:.2f} degrees")

# Select and Segment the planar points
mask = np.ones(len(points), dtype=bool)
mask[plane_inliers] = False
outliers = points[mask]

plane_cloud = o3d.geometry.PointCloud()
plane_cloud.points = o3d.utility.Vector3dVector(points[plane_inliers])
plane_cloud.paint_uniform_color([0.8, 0.2, 0.6])

plane_outliers = o3d.geometry.PointCloud()
plane_outliers.points = o3d.utility.Vector3dVector(outliers)
plane_outliers.paint_uniform_color([0.3, 0.1, 0.9])

o3d.visualization.draw_geometries([plane_cloud, plane_outliers])