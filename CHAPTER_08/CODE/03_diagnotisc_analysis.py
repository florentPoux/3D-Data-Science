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
#%%

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#%% Styling prompt

plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 600

#Optional Configuration and package version
import sys
print(sys.version)


#%% Data Paths definition

point_cloud_file = "../DATA/office.ply"
plane_file = "../DATA/ground_plane.ply"

#%% Definition of functions

#Histogram
def hist_styled(feature, f_name):
    # Create histogram
    n, bins, patches = plt.hist(feature, bins=100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=4)
    n = n.astype('int') # it MUST be integer
    
    # Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.cool(1-n[i]/max(n)))

    
    # Calculate statistics
    mean_val = np.mean(feature)
    median_val = np.median(feature)
    mode_val = bins[np.argmax(n)]
    
    # Add vertical lines for mean, median, and mode
    plt.axvline(mean_val, color='lightcoral', linestyle='dashed', linewidth=1.3, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='palegreen', linestyle='dashed', linewidth=1.3, label=f'Median: {median_val:.3f}')
    plt.axvline(mode_val, color='skyblue', linestyle='dashed', linewidth=1.3, label=f'Mode: {mode_val:.3f}')
    
    # Set title and labels
    plt.title(f'{f_name}-feature: Distribution', fontsize=10)
    plt.xlabel(f'{f_name} values', fontsize=10)
    plt.ylabel('Point Numbers', fontsize=10)
    
    # Add grid
    plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)
    
    # Add legend
    plt.legend(fontsize=7)
    
    plt.show()
    return

#Distance to plane

def compute_distance_to_plane(point_cloud_file, plane_file):
    # Load point cloud and plane mesh
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    plane_mesh = o3d.io.read_triangle_mesh(plane_file)

    # Compute distances from point cloud to plane
    distances = []
    for point in point_cloud.points:
        min_distance = np.inf
        for i, triangle in enumerate(plane_mesh.triangles):
            v0 = plane_mesh.vertices[triangle[0]]
            v1 = plane_mesh.vertices[triangle[1]]
            v2 = plane_mesh.vertices[triangle[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            distance = np.abs(np.dot(point - v0, normal))
            min_distance = min(min_distance, distance)
        distances.append(min_distance)

    distances = np.array(distances)

    # Create a colored point cloud based on distance to plane
    point_cloud.colors = o3d.utility.Vector3dVector(plt.cm.cool(distances / np.max(distances))[:, :3])

    # Export the distance histogram
    plt.figure()
    plt.hist(distances, bins=50, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=4)    

    plt.xlabel("Distance to plane")
    plt.ylabel("Frequency")
    plt.savefig("../RESULTS/distance_histogram.png")

    return point_cloud, plane_mesh, distances


#%% Funciton execution and visualization

colored_point_cloud, plane_mesh, distances = compute_distance_to_plane(point_cloud_file, plane_file)

# hist_styled(distances, "Distance to plane")
# Visualize the colored point cloud
o3d.visualization.draw_geometries([colored_point_cloud, plane_mesh])

#%% Distance to 3D Mesh Funciton definition

def compute_distance_to_mesh(point_cloud_file, mesh_file):
    # Load point cloud and mesh
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # Compute KDTree for the mesh
    # mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=1000000)
    mesh_tree = o3d.geometry.KDTreeFlann(pcd_mesh)

    # Compute distances from point cloud to mesh
    distances = []
    for point in point_cloud.points:
        _, inds, dist = mesh_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))

    distances = np.array(distances)

    # Create a colored point cloud based on distance to mesh
    point_cloud.colors = o3d.utility.Vector3dVector(plt.cm.cool(distances / np.max(distances))[:, :3])

    # Export the distance histogram
    plt.figure()
    plt.hist(distances, bins=50, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=4)
    plt.xlabel("Distance to mesh")
    plt.ylabel("Frequency")
    plt.savefig("distance_histogram.png")

    return point_cloud, mesh, distances

#%% Executing the funciton on two new datasets

point_cloud_file = "../DATA/pcd_car.ply"
mesh_file = "../DATA/mesh_car.ply"
colored_point_cloud, mesh, distances = compute_distance_to_mesh(point_cloud_file, mesh_file)

# Visualize the colored point cloud
o3d.visualization.draw_geometries([colored_point_cloud, mesh])