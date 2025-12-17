"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 08

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
#%% Libraries import

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

#%% Dataset

# Load a point cloud (ensure it has intensity information)
pcd = o3d.io.read_point_cloud("../DATA/chair_colored_notpure.ply")
o3d.visualization.draw_geometries([pcd])

#%% 1. RGB Color Analysis

def analyze_rgb(pcd):
    colors = np.asarray(pcd.colors)
    
    # Color histogram
    plt.figure(figsize=(15, 5))
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.subplot(1, 3, i+1)
        plt.hist(colors[:, i], bins=50, color=color, alpha=0.7)
        plt.title(f'{color.capitalize()} Channel Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Color-based segmentation (simple example)
    red_mask = colors[:, 0] > 0.1  # Segment points with high red values
    red_cloud = pcd.select_by_index(np.where(red_mask)[0])
    o3d.visualization.draw_geometries([red_cloud])

analyze_rgb(pcd)

#%% 2. Surface Normals Analysis

def analyze_normals(pcd, radius=0.1):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)

    # Visualize normal consistency
    consistencies = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
        neighborhood_normals = normals[idx[1:]]
        consistency = np.mean(np.abs(np.dot(neighborhood_normals, normals[i])))
        consistencies.append(consistency)

    plt.hist(consistencies, bins=50)
    plt.title('Normal Consistency Distribution')
    plt.xlabel('Consistency')
    plt.ylabel('Frequency')
    plt.show()

    # Visualize point cloud with normals
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

analyze_normals(pcd)

#%% 3. EigenValue Analysis

def compute_eigenvalue_features(pcd, radius=0.1):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    eigenvalues = []

    for point in points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
        if k < 4:  # Need at least 4 points to compute meaningful eigenvalues
            eigenvalues.append([0, 0, 0])
            continue
        neighborhood = points[idx[1:]]  # Exclude the point itself
        covariance = np.cov(neighborhood.T)
        eigvals = np.linalg.eigvals(covariance)
        eigenvalues.append(sorted(eigvals, reverse=True))

    eigenvalues = np.array(eigenvalues)
    
    # Compute linearity, planarity, and sphericity
    linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / eigenvalues[:, 0]
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / eigenvalues[:, 0]
    sphericity = eigenvalues[:, 2] / eigenvalues[:, 0]

    # Visualize features
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.hist(linearity, bins=50)
    plt.title('Linearity')
    plt.subplot(132)
    plt.hist(planarity, bins=50)
    plt.title('Planarity')
    plt.subplot(133)
    plt.hist(sphericity, bins=50)
    plt.title('Sphericity')
    plt.tight_layout()
    plt.show()

    return linearity, planarity, sphericity

linearity, planarity, sphericity = compute_eigenvalue_features(pcd, 5)

#%% Visualize features on point cloud
features = [linearity, planarity, sphericity]
feature_names = ['Linearity', 'Planarity', 'Sphericity']

for feature, name in zip(features, feature_names):
    feature_pcd = o3d.geometry.PointCloud()
    feature_pcd.points = pcd.points
    
    colors = cm.viridis(feature)[:, :3]
    
    feature_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([feature_pcd], window_name=f"{name} Visualization")


#%% 4. Intensity Analysis

def analyze_intensity(pcd):
    # Assuming intensity is stored in the 'colors' attribute as grayscale
    intensities = np.asarray(pcd.colors)[:, 0]  # Use red channel as intensity

    # Visualize intensity distribution
    plt.hist(intensities, bins=50)
    plt.title('Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()

    # Simple intensity-based segmentation
    high_intensity_mask = intensities > 0.8
    high_intensity_cloud = pcd.select_by_index(
        np.where(high_intensity_mask)[0])

    # Visualize high intensity points
    high_intensity_cloud.paint_uniform_color(
        [1, 0, 0])  # Color red for visibility
    o3d.visualization.draw_geometries([high_intensity_cloud])

    return intensities

intensities = analyze_intensity(pcd)