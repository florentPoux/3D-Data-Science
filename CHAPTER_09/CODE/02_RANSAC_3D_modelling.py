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

#%% Application 3: 3D Modelling

#%% 1. Libraries import

import numpy as np
import open3d as o3d

from sklearn.neighbors import KDTree

#%% 2. Dataset import

pcd = o3d.io.read_point_cloud("../DATA/pcd_synthetic.ply")

points = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])

#%% 3. Automatic Threshold Computation

tree = KDTree(points, leaf_size=2)
nearest_dist, nearest_ind = tree.query(points, k=8)
nearest_dist_mean = np.mean(nearest_dist[:,1:],axis=0)
threshold = np.min(nearest_dist_mean) + np.std(nearest_dist_mean)

#%% 4. RANSAC Function Definition

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

def ransac_sphere(points, num_iterations=1000, threshold=0.1):
    best_inliers = []
    best_sphere = None
    
    for _ in range(num_iterations):
        # Randomly sample 4 points
        sample = points[np.random.choice(points.shape[0], 4, replace=False)]
        
        # Calculate sphere equation (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
        A = np.array([
            [2*(sample[1][0]-sample[0][0]), 2*(sample[1][1]-sample[0][1]), 2*(sample[1][2]-sample[0][2])],
            [2*(sample[2][0]-sample[0][0]), 2*(sample[2][1]-sample[0][1]), 2*(sample[2][2]-sample[0][2])],
            [2*(sample[3][0]-sample[0][0]), 2*(sample[3][1]-sample[0][1]), 2*(sample[3][2]-sample[0][2])]
        ])
        
        B = np.array([
            [sample[1][0]**2 + sample[1][1]**2 + sample[1][2]**2 - sample[0][0]**2 - sample[0][1]**2 - sample[0][2]**2],
            [sample[2][0]**2 + sample[2][1]**2 + sample[2][2]**2 - sample[0][0]**2 - sample[0][1]**2 - sample[0][2]**2],
            [sample[3][0]**2 + sample[3][1]**2 + sample[3][2]**2 - sample[0][0]**2 - sample[0][1]**2 - sample[0][2]**2]
        ])
        
        center = np.linalg.solve(A, B).flatten()
        radius = np.sqrt(np.sum((sample[0] - center)**2))
        
        # Calculate distances of all points to the sphere surface
        distances = np.abs(np.sqrt(np.sum((points - center)**2, axis=1)) - radius)
        
        # Count inliers
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sphere = (*center, radius)
    
    return best_sphere, best_inliers

#%% 5. Apply RANSAC Shape Detection

# Find plane and sphere using RANSAC
plane_params, plane_inliers = ransac_plane(points, 1000, threshold)
sphere_params, sphere_inliers = ransac_sphere(points, 1000, 0.1)

print(f"Plane equation: {plane_params[0]}x + {plane_params[1]}y + {plane_params[2]}z + {plane_params[3]} = 0")
print(f"Sphere equation: (x - {sphere_params[0]})^2 + (y - {sphere_params[1]})^2 + (z - {sphere_params[2]})^2 = {sphere_params[3]**2}")

#%% 6. Segment the dataset into planar and spherical points


# Select and Segment the planar points
plane_cloud = o3d.geometry.PointCloud()
plane_cloud.points = o3d.utility.Vector3dVector(points[plane_inliers])
plane_cloud.paint_uniform_color([1, 0, 0])  # Red
o3d.visualization.draw_geometries([plane_cloud])

# Select and Segment the spherical points
sphere_cloud = o3d.geometry.PointCloud()
sphere_cloud.points = o3d.utility.Vector3dVector(points[sphere_inliers])
sphere_cloud.paint_uniform_color([0, 1, 0])  # Green
o3d.visualization.draw_geometries([sphere_cloud])

#%% 7. Make sure to clear the outliers

from functools import reduce

# Select and Segment the remaining points
other_points = np.delete(points, reduce(np.union1d, (plane_inliers, sphere_inliers)), axis=0)

other_cloud = o3d.geometry.PointCloud()
other_cloud.points = o3d.utility.Vector3dVector(other_points)
other_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Dark

# Visualize
o3d.visualization.draw_geometries([plane_cloud, sphere_cloud, other_cloud])

#%% 8. Create the functions to generate 3D Geometric Shapes

def create_plane_mesh(plane_params, size=2, resolution=20):
    a, b, c, d = plane_params
    
    # Create a grid of points on the plane
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z coordinates
    Z = (-d - a*X - b*Y) / c
    
    # Create vertices and triangles
    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    triangles = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            triangles.extend([[v0, v2, v1], [v1, v2, v3]])
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

def create_sphere_mesh(sphere_params, resolution=50):
    center_x, center_y, center_z, radius = sphere_params
    
    # Create a UV sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center_x + radius * np.outer(np.cos(u), np.sin(v))
    y = center_y + radius * np.outer(np.sin(u), np.sin(v))
    z = center_z + radius * np.outer(np.ones_like(u), np.cos(v))
    
    # Create vertices and triangles
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    triangles = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            triangles.extend([[v0, v2, v1], [v1, v2, v3]])
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

#%% 9. Create the 3D mesh representations

plane_mesh = create_plane_mesh(plane_params)
plane_mesh.paint_uniform_color([1, 0.7, 0.7])  # Light red

sphere_mesh = create_sphere_mesh(sphere_params)
sphere_mesh.paint_uniform_color([0.7, 1, 0.7])  # Light green


# Visualize
o3d.visualization.draw_geometries([plane_cloud, sphere_cloud, other_cloud, plane_mesh, sphere_mesh],mesh_show_back_face = True)

#%% 10. Optimizing: Creating an Oriented Plane Function

def create_oriented_plane_mesh(plane_params, inlier_points):
    a, b, c, d = plane_params
    normal = np.array([a, b, c])
    
    # Project inlier points onto the plane
    projected_points = inlier_points - (np.dot(inlier_points, normal) + d).reshape(-1, 1) * normal / np.dot(normal, normal)
    
    # Perform PCA to find the principal directions on the plane
    mean = np.mean(projected_points, axis=0)
    centered_points = projected_points - mean
    _, _, vh = np.linalg.svd(centered_points)
    
    # The first two right singular vectors are the principal directions
    u = vh[0]
    v = vh[1]
    
    # Calculate the extent of the points along the principal directions
    u_coords = np.dot(centered_points, u)
    v_coords = np.dot(centered_points, v)
    u_range = np.max(u_coords) - np.min(u_coords)
    v_range = np.max(v_coords) - np.min(v_coords)
    
    # Create a grid of points on the plane
    resolution = 20
    u_grid = np.linspace(-u_range/2, u_range/2, resolution)
    v_grid = np.linspace(-v_range/2, v_range/2, resolution)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Calculate the 3D coordinates of the grid points
    grid_points = mean + U.reshape(-1, 1) * u + V.reshape(-1, 1) * v
    
    # Create vertices and triangles
    vertices = grid_points
    triangles = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            triangles.extend([[v0, v2, v1], [v1, v2, v3]])
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

#%% 11. Create the final mesh representations

plane_mesh = create_oriented_plane_mesh(plane_params,points[plane_inliers])
plane_mesh.paint_uniform_color([1, 0.7, 0.7])  # Light red

sphere_mesh = create_sphere_mesh(sphere_params)
sphere_mesh.paint_uniform_color([0.7, 1, 0.7])  # Light green

# Visualize
o3d.visualization.draw_geometries([plane_cloud, sphere_cloud, other_cloud, plane_mesh, sphere_mesh],mesh_show_back_face = True)