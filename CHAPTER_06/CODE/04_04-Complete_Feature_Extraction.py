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
## Chapter 4 - 4: End to End Feature Extraction

General Information
* Created by: 🦊 Florent Poux. 
* Copyright: Florent Poux.
* License: MIT
* Status: Review Only (Confidential)

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Have fun with this notebook.\
You can very simply run in a coherent environment (ctrl+Enter for each cell) !

🎵 Note: Styling was not taken care of at this stage.

Enjoy!
"""

#%% 0. Gathering Data

#%% 1. Python Environment and Library setup

#Base Libraries
import numpy as np
import time

# [OPTIONAL] Project Module
from scipy.spatial import KDTree


# 3D Library
import pyvista as pv

#%% 2. Data Loading and Fundamentals (PyVista)

#%% 2.1. I/O Operations

pcd_pv = pv.read('../DATA/features_verviers.ply')

#Plot quickly with EDL on
pcd_pv.plot(eye_dome_lighting=True, rgb=True)



#%% 2.2. Exploring PyVista Capabilities

#Storing in variable for pyvista
pcd_pv['elevation'] = pcd_pv.points[:,2]
pcd_pv['random'] = pcd_pv.points[:,0] * pcd_pv.points[:,1]
#Render as spheres
pv.plot(pcd_pv, scalars = pcd_pv['elevation'], render_points_as_spheres=True, point_size=5, show_scalar_bar=False)


#%% 3. Pre-Processing: Creating a 3D Data Structure

#%% 3.1. KD-Tree with PyVista (limitations)

#one point at a time not efficient
temp = pcd_pv.find_closest_point((1, 1, 0), n = 20)
print(temp)

#%% 3.2. K-Nearest neighbor structure (SciPy)

# Build a KD-trees for each point cloud
tree = KDTree(pcd_pv.points)

#%% 3.3. K-Nearest neighbor search and indexing

t0 = time.time()

# Find nearest neighbors for each point in the point cloud
dists, indices = tree.query(pcd_pv.points, k=50)

# Get the neighbor points for each point

neighbors = pcd_pv.points[indices]

t1 = time.time()
print(f"Neihgbor Computation in {t1-t0} seconds")

#%% 4. Point Cloud Data Featuring: PCA

#%% 4.1. Compute PCA for a subcloud

X = pcd_pv.points

# Compute the mean of the data
mean = np.mean(X, axis=0)

# Center the data by subtracting the mean
centered_data = X - mean

# Compute the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Get the eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Sort the eigenvectors by decreasing eigenvalues
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

print(' sorted_index: ', sorted_index, '\n sorted_eigenvalue: ', sorted_eigenvalue, '\n sorted_eigenvectors: \n', sorted_eigenvectors)


#%% 4.2. Define the PCA computation function

sel = 2

def PCA(cloud):
    mean = np.mean(cloud, axis=0)
    centered_data = cloud - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    return sorted_eigenvalue, sorted_eigenvectors
#Single unit test
f_val, f_vec = PCA(neighbors[sel])

print(' sorted_eigenvalue: ', f_val, '\n sorted_eigenvectors: \n', f_vec)


#%% 4.3. Define PCA_based Features Computation Function

def PCA_featuring(val, vec):
    planarity = (val[1] - val[2]) / val[0]
    linearity = (val[0] - val[1]) / val[0]
    omnivariance = (val[0] * val[1] * val[2]) ** (1/3)
    _, _, normal = vec
    verticality = 1 - abs(normal[2])
    return planarity, linearity, omnivariance, verticality, normal[0], normal[1], normal[2]

#test
p, l, o, v, nx, ny, nz = PCA_featuring(f_val, f_vec)
print('', p, '\n', l, '\n', o, '\n', v, '\n', nx, '\n', ny, '\n', nz)

#%% 5. Understanding Feature Neighborhood Definition


#%% 5.1. K-NN Search
tree_temp = KDTree(pcd_pv.points)
t0 = time.time()
_, idx_temp = tree_temp.query(pcd_pv.points, k=20)
t1 = time.time()
print(f"KNN Search Computation in {t1-t0} seconds")

#%% 5.2. Radius Search
t0 = time.time()
tree_temp = KDTree(pcd_pv.points)
idx_temp = tree_temp.query_ball_point(pcd_pv.points, 0.2)
t1 = time.time()
print(f"Radius Search Search Computation in {t1-t0} seconds")

#%% 5.3. Knowledge Driven Custom Search
t0 = time.time()
tree_2D = KDTree(pcd_pv.points[:,0:2])
idx_2D_rad = tree_2D.query_ball_point(pcd_pv.points[:,0:2], 0.2)
t1 = time.time()
print(f"Knowledge Driven Custom Search Computation in {t1-t0} seconds")

#%% 6. Point Cloud Feature Extraction: Relative Featuring

t0 = time.time()
sel = 1
selection = pcd_pv.points[idx_2D_rad[sel]]

d_high = np.array(np.max(selection, axis = 0) - pcd_pv.points[sel])[2]
d_low = np.array(pcd_pv.points[sel] - np.min(selection, axis = 0))[2]

#%% 7. Automation for Full Dataset

#%% 7.1. Prepare feature matrix

dt = {'names':['planarity', 'linearity', 'omnivariance', 'verticality', 'nx', 'ny', 'nz', 'd_high', 'd_low'], 'formats':[float, float, float, float, float, float, float, float, float]}

features = np.empty(len(pcd_pv.points), dtype=dt)
features[:] = np.nan

#%% 7.2. Loop and automate

pts_number = len(pcd_pv.points)

#1. We have the neighbors per point
t0 = time.time()
for idx in range(pts_number):
    f_val, f_vec = PCA(neighbors[idx])
    features['planarity'][idx], features['linearity'][idx], features['omnivariance'][idx], features['verticality'][idx], features['nx'][idx], features['ny'][idx], features['nz'][idx] = PCA_featuring(f_val, f_vec)
    if len(idx_2D_rad[idx])>2:
        selection = pcd_pv.points[idx_2D_rad[idx]]
        features['d_high'][idx] = np.array(np.max(selection, axis = 0) - pcd_pv.points[idx])[2]
        features['d_low'][idx] = np.array(pcd_pv.points[idx] - np.min(selection, axis = 0))[2]
t1 = time.time()

print(f"Full Point Cloud Feature Computation in {t1-t0} seconds")

#%% 8. Visualization (Pyvista)

# pv.plot(pcd_pv, scalars = features['d_high'], render_points_as_spheres=True, point_size=5, show_scalar_bar=False)
# pv.plot(pcd_pv, scalars = features['planarity'], render_points_as_spheres=True, point_size=5, show_scalar_bar=False)
pv.plot(pcd_pv, scalars = features['verticality'], render_points_as_spheres=True, point_size=5, show_scalar_bar=False)
# pv.plot(pcd_pv, scalars = features['omnivariance'], render_points_as_spheres=True, point_size=5, show_scalar_bar=True)


#%% 8.1. Interactive Selection: 

#N Using the Scalars
pcd_pv['d_high'] = features['d_high']
pcd_pv['d_low'] = features['d_low']
pcd_pv['verticality'] = features['verticality']
pcd_pv['planarity'] = 1- features['planarity']
# Set up a plotter
p = pv.Plotter()

# add the interactive thresholding tool and show everything
p.add_mesh_threshold(pcd_pv, scalars = 'planarity', title = 'planarity Pointer (%)', all_scalars = True, render_points_as_spheres=True, point_size=7)
p.show()


#%% 8.2. Interactive Selection and Remanence

#N Using the Scalars
pcd_pv['d_high'] = features['d_high']
pcd_pv['d_low'] = features['d_low']

# Set up a plotter
p = pv.Plotter()
# add the interactive thresholding tool and show everything
p.add_mesh_threshold(pcd_pv, scalars ='d_high', title = 'distance (m)', all_scalars = True, render_points_as_spheres=True, point_size=7)
p.enable_mesh_picking(pcd_pv)
p.show()


#%% 8.3. Selection refinement
selection = p.picked_mesh
pcd_pv['planarity'] = features['planarity']

p2 = pv.Plotter()
# add the interactive thresholding tool and show everything
p2.add_mesh_threshold(selection, scalars ='planarity', title = 'planarity (%)', all_scalars = True, render_points_as_spheres=True, point_size=7)
# p2.enable_mesh_picking(selection)
p2.show()



#%% 8.4. Selection export

selection_path = '../RESULTS/segmentation_live.obj'

selection = p.picked_mesh
pl = pv.Plotter()
_ = pl.add_mesh(selection)
pl.export_obj(selection_path)

#%% 8.5. Load and test visualization

#%% 8.5. Load and test visualization

# Read the dataset point cloud
pcd_pv2 = pv.read(selection_path)

#Plot quickly with EDL on
pcd_pv2.plot(eye_dome_lighting=True)


#%% 9. Merge with coordinates and save the results

import numpy.lib.recfunctions as rfn

dt2 = {'names':['X', 'Y', 'Z'], 'formats':[float, float, float]}
coords = np.empty(len(pcd_pv.points), dtype=dt2)
coords['X'] = pcd_pv.points[:,0]
coords['Y'] = pcd_pv.points[:,1]
coords['Z'] = pcd_pv.points[:,2]
pcd_featured = rfn.merge_arrays((coords, features), flatten=True)

np.savetxt('../RESULTS/pcd_with_features.txt', pcd_featured, fmt='%.5f', delimiter=',', header=','.join((dt2['names'] + dt['names'])))


#%% 10. Application to Ground and Tree Classification (CloudCompare)

#%% [OPTIONAL BONUS IS BELOW]

#%% Multiple sliders

from pyvistaqt import BackgroundPlotter

class MyCustomRoutine:
    def __init__(self, mesh):
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            'radius': 0.5,
            'theta_resolution': 30,
            'phi_resolution': 30,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        result = pv.Sphere(**self.kwargs)
        self.output.copy_from(result)
        return

engine = MyCustomRoutine(pcd_pv)

p = pv.Plotter()
# add the interactive thresholding tool and show everything
p.add_mesh_threshold(pcd_pv, scalars ='area', title = 'area (m)', all_scalars = True, render_points_as_spheres=True, point_size=7)
p.enable_mesh_picking(pcd_pv)




p.add_slider_widget(
    callback=lambda value: engine('elevation', int(value)),
    rng=[3, 60],
    value=30,
    title="Phi Resolution",
    pointa=(0.025, 0.1),
    pointb=(0.31, 0.1),
    style='modern',
)
p.add_slider_widget(
    callback=lambda value: engine('area', int(value)),
    rng=[3, 60],
    value=30,
    title="Theta Resolution",
    pointa=(0.35, 0.1),
    pointb=(0.64, 0.1),
    style='modern',
)
# p.add_slider_widget(
#     callback=lambda value: engine('radius', value),
#     rng=[0.1, 1.5],
#     value=0.5,
#     title="Radius",
#     pointa=(0.67, 0.1),
#     pointb=(0.98, 0.1),
#     style='modern',
# )
p.show()
 
#%% Export as html results

import pyvista as pv
from pyvista import examples
mesh = examples.load_uniform()
pl = pv.Plotter(shape=(1, 2))
_ = pl.add_mesh(
    mesh, scalars='Spatial Point Data', show_edges=True
)
pl.subplot(0, 1)
_ = pl.add_mesh(
    mesh, scalars='Spatial Cell Data', show_edges=True
)
pl.export_html('pv.html')  

#%% Connectivity filtering

# Use the connectivity filter to get the scalar array of region ids
conn = pcd_pv.connectivity(largest=False)
# See the active scalar fields in the calculated object
print(conn.active_scalars_name)
# Show the ids
print(conn["RegionId"])
# Set up a plotter
p = pv.Plotter()
# add the interactive thresholding tool and show everything
p.add_mesh_threshold(conn,show_edges=True)
p.show()

#%% 6. Define something that builds itself

# Class on building a replica point cloud one point neighbourhood at a time. 
class find_neighbours():
    def __init__(self,pc_points, num_neighbours):
        # Setup initial variables - get the point cloud, set a counter for the current point selected, getting the neighborhood
        # setup a array of booleans to keep track of the points that have been already checked and setting up the plotter
        self.pc_points = pc_points
        self.counter = 0
        self.num_neighbours = num_neighbours
        self.checked = np.zeros(len(pc_points), dtype=bool)
        self.p = BackgroundPlotter()


    # Update function called by the callback
    def every_point_neighborhood(self):
        # get the current point to calculate the neighborhood of
        point = pc_points[self.counter,:]
        # get all the indices of the neighbors of the current point
        index = pc.find_closest_point(point,self.num_neighbours)
        # get the neighbor points
        neighbours = pc_points[index,:]
        # mark the points as checked and extract the checked sub-point cloud
        self.checked[index] = True

        new_pc = pc_points[self.checked]
        # move the reconstructed point cloud in X direction so it can be more easier seen
        new_pc[:,0]+=1
        # add the neighborhood points, the center point and the new checked point clouds to the plotter.
        # Because we are using the same names PyVista knows to update the already existing ones
        self.p.add_mesh(neighbours, color="r", name='neighbors', point_size=8.0, render_points_as_spheres=True)
        self.p.add_mesh(point, color="b", name='center', point_size=10.0, render_points_as_spheres=True)
        self.p.add_mesh(new_pc, color="g", name='new_pc', render_points_as_spheres=True)
        # move the counter with a 100 points so the visualization is faster - change this to 1 to get all points
        self.counter+=100
        # get the point count
        pc_count = len(pc_points)
        # check if all points have been done. If yes then 0 the counter and the checked array
        if self.counter >= pc_count:
            self.counter = 0
            self.checked = np.zeros(len(pc_points), dtype=bool)

        # We update the whole plotter
        self.p.update()
    # visualization function
    def visualize_neighbours(self):
        # add the colored mesh of the duck statue. We set the RGB color scalar array as color by calling rgb=True
        self.p.add_mesh(pc, render_points_as_spheres=True, rgb=True)
        # We set the callback function and an interval of 100 between update cycles
        self.p.add_callback(self.every_point_neighborhood, interval=100)
        self.p.show()
        self.p.app.exec_()

#%% 7. Find and Visualize neighbors

pc = pv.read(('../DATA/MLS_UTWENTE_super_sample.ply')) 
pc_points = pc.points
pc_points = np.array(pc_points)

neighbours_class = find_neighbours(pc_points, 400)
neighbours_class.visualize_neighbours()

#%% 8. Manipulation

# Callback function for selecting points with the mouse
def manipulate_picked(point):
    # Get selected point and switch the boolean for having selected something to True
    params.point = point
    params.point_selected = True
    # Get the closest points indices 
    index = pc.find_closest_point(point,params.size)
    # Get the points themselves
    neighbours = pc_points[index,:]
    # add points representing the neighborhood and the selected point to the plotter
    p.add_mesh(neighbours, color="r", name='neighbors', point_size=8.0, render_points_as_spheres=True)
    p.add_mesh(point, color="b", name='center', point_size=10.0, render_points_as_spheres=True)

# Callback function for the slider widget for changing the neighborhood size 
def change_neighborhood(value):
    # change the slider value to int
    params.size = int(value)
    # call the point selection function if a point has already been selected
    if params.point_selected:
        manipulate_picked(params.point)

    return

# Class of parameters used in the two callback function
class params():
    # size of the neighborhood 
    size = 200
    # the selected point
    point = np.zeros([1,3])
    # is the point selected or not
    point_selected = False

pc = pv.read(('../DATA/MLS_UTWENTE_super_sample.ply')) 
# Initialize the plotter
p = pv.Plotter()
# Add the main duck statue point cloud as spheres and with colors 
p.add_mesh(pc, render_points_as_spheres=True, rgb=True)
# Initialize the mouse point picker callback and the slider widget callback
p.enable_point_picking(callback=manipulate_picked,show_message="Press left mouse to pick", left_clicking=True)
p.add_slider_widget(change_neighborhood, [10, 300], 20, title='Neighbourhood size')
p.show()

#%%