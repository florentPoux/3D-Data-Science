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
# O'Reilly: 3D Data Science with Python
## Chapter 4 - 3: 3D Data Registration

General Information
* Created by: 🦊 Florent Poux. 
* Copyright: Florent Poux.
* License: MIT
* Status: Review Only (Confidential)

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Enjoy!
"""

#%% 1. Library imports

#Bse library
import numpy as np

#3D Library
import open3d as o3d

import copy


#%% Load point clouds

source = o3d.io.read_point_cloud("../DATA/registration_source.ply")
target = o3d.io.read_point_cloud("../DATA/global_todo_registration_target.ply")

o3d.visualization.draw_geometries([source, target])

#%% Definition of Functions

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def prepare_dataset(source, target, voxel_size):
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

#%% Pre-processing

# Set voxel size for downsampling
voxel_size = 1.5

# Prepare dataset
source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

#%% Perform global registration
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

print("Global registration result:")
print(result_ransac)
print("Transformation matrix:")
print(result_ransac.transformation)

#%% Visualize
source_global = copy.deepcopy(source)
source_global.transform(result_ransac.transformation)

o3d.visualization.draw_geometries([source_global, target])


#%% The ICP Registration Definition

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

#%% Normal computation for not downsampled point cloud

source.estimate_normals(
search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
target.estimate_normals(
search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

o3d.visualization.draw_geometries([source, target])



#%% Refine the registration using ICP
result_icp = refine_registration(source, target, result_ransac, voxel_size)
print("Local registration (ICP) result:")
print(result_icp)
print("Refined transformation matrix:")
print(result_icp.transformation)

#%% Visualize
# Transform the source point cloud
source_local = copy.deepcopy(source)
source_local.transform(result_icp.transformation)

# Visualize the result
o3d.visualization.draw_geometries([source_local, target])