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

#%% 1. Library Import

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

#%% 2. I/O + Viz'

point_clouds = ["../DATA/darth_vader.ply", "../DATA/depth_anything.ply", "../DATA/industrial_room_part.ply", "../../DATA/NAAVIS_EXTERIOR.ply"]

for i in point_clouds:
    pcd = o3d.io.read_point_cloud(i)
    o3d.visualization.draw_geometries([pcd])

# Convert Open3d Object to Numpy for processing stages
input_points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.asarray(pcd.colors)

#%% 3. Parameter Set Up

angle_threshold = np.pi/6
distance_threshold = 0.1 # (NAAVIS)
# distance_threshold = 0.01 # Industrial

#%% 4. Region Growing Initialization

N = len(input_points) #Total Number of Points
kd_tree = KDTree(input_points)
segments = [] # Output of the Algorithm
unsegmented = set(range(N)) # Algorithm Worker

#%% 5. Segmentation Loop (Normals)

import time

t0 = time.time()
while unsegmented:
    # Start a New Segment
    seed_index = unsegmented.pop()
    segment = [seed_index]
    # stack of points to be processed (neighbor indices)
    stack = list(kd_tree.query_ball_point(input_points[seed_index], distance_threshold))

    while stack:
        point_index = stack.pop()
        if point_index not in unsegmented:
            continue # Makes sure that we consider "free" points

        normal_angle = np.abs(np.arccos(np.dot(normals[seed_index], normals[point_index])))
        if normal_angle < angle_threshold:
            unsegmented.remove(point_index)
            segment.append(point_index)
            stack.extend(kd_tree.query_ball_point(input_points[point_index], distance_threshold))

    segments.append(segment)
t1 = time.time()

print(f"Region Growing Successful in: {round(t1-t0,3)} seconds")

#%% 6. Assigning a Label + Color per Segment

def coloring_segments(input_points, segments):
    colors_l = np.zeros_like(input_points)
    labels = np.zeros(len(input_points), dtype=int)
    
    count = 0
    for segment in segments:
        # color = np.random.rand(3)  # Generate a random color
          # Assign the color to all points in the segment
        # Optional
        if len(segment) <= 1:
            labels[segment] = -1
            colors_l[segment] = [0,0,0]
            count-=1
        elif len(segment) > 1 and len(segment) < 10:
            labels[segment] = 0
            colors_l[segment] = [1,0,0]
            count-=1            
        else:
            labels[segment] = count
            colors_l[segment] = np.random.rand(3)
        count+=1
    return labels, colors_l

labels, colors_l = coloring_segments(input_points, segments)

#%% 7. Write Results to Variables
o3d.visualization.draw_geometries([pcd])
pcd.colors = o3d.utility.Vector3dVector(colors_l)
pcd_segmented = np.hstack((input_points, np.atleast_2d(labels).T))

#Qualitative Analysis
o3d.visualization.draw_geometries([pcd])

#%% 8. Export to .PLY + .ASCII File for downward processes

o3d.io.write_point_cloud("../RESULTS/NAAVIS_EXTERIOR_segmented.ply", pcd)
np.savetxt("../RESULTS/NAAVIS_EXTERIOR_segmented.xyz", pcd_segmented, fmt='%1.6f', delimiter=';', header='X;Y;Z;Segment')


#%% 9. Color-based Segmentation Function (Color)

import time
from skimage.color import rgb2lab, deltaE_ciede2000

def color_similarity(color1, color2):

  lab1 = rgb2lab(color1)
  lab2 = rgb2lab(color2)
  return deltaE_ciede2000(lab1, lab2)

def rg_color(input_points, colors, distance_threshold, c_threshold = 10):
    # Initialization
    N = len(input_points) #Total Number of Points
    kd_tree = KDTree(input_points)
    segments = [] # Output of the Algorithm
    unsegmented = set(range(N)) # Algorithm Worker

    while unsegmented:
        # Start a New Segment
        seed_index = unsegmented.pop()
        segment = [seed_index]
        # stack of points to be processed (neighbor indices)
        stack = list(kd_tree.query_ball_point(input_points[seed_index], distance_threshold))
    
        while stack:
            point_index = stack.pop()
            if point_index not in unsegmented:
                continue # Makes sure that we consider "free" points
    
            c_dist = color_similarity(colors[seed_index], colors[point_index])
            # print(c_dist)
            if c_dist < c_threshold:
                unsegmented.remove(point_index)
                segment.append(point_index)
                stack.extend(kd_tree.query_ball_point(input_points[point_index], distance_threshold))
    
        segments.append(segment)
    return segments

#%% 10. Color-based Segmentation 

pcd = o3d.io.read_point_cloud(i)

# Convert Open3d Object to Numpy for processing stages
input_points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.asarray(pcd.colors)

t0 = time.time()
segments_color = rg_color(input_points, colors, 0.1, 20)
t1 = time.time()

print(f"Region Growing Successful in: {round(t1-t0,3)} seconds")

#%% 11. Visualizing the results

labels, colors_l = coloring_segments(input_points, segments_color)

pcd.colors = o3d.utility.Vector3dVector(colors_l)
pcd_segmented = np.hstack((input_points, np.atleast_2d(labels).T))

#Qualitative Analysis
o3d.visualization.draw_geometries([pcd])

#%% 12. Iterating Tests

def rg_normals(input_points, normals, distance_threshold = 0.1, angle_threshold = np.pi/6):
    # Initialization
    N = len(input_points) #Total Number of Points
    kd_tree = KDTree(input_points)
    segments = [] # Output of the Algorithm
    unsegmented = set(range(N)) # Algorithm Worker
    
    # Segmentation Loop (Normal)
    
    while unsegmented:
        # Start a New Segment
        seed_index = unsegmented.pop()
        segment = [seed_index]
        # stack of points to be processed (neighbor indices)
        stack = list(kd_tree.query_ball_point(input_points[seed_index], distance_threshold))
    
        while stack:
            point_index = stack.pop()
            if point_index not in unsegmented:
                continue # Makes sure that we consider "free" points
    
            normal_angle = np.abs(np.arccos(np.dot(normals[seed_index], normals[point_index])))
            if normal_angle < angle_threshold:
                unsegmented.remove(point_index)
                segment.append(point_index)
                stack.extend(kd_tree.query_ball_point(input_points[point_index], distance_threshold))
    
        segments.append(segment)
    return segments

t0 = time.time()
segments_n = rg_normals(input_points, normals, 0.1, np.pi/2)
t1 = time.time()
print(f"Region Growing Successful in: {round(t1-t0,3)} seconds")
print(len(segments_n))
_ , colors_l = coloring_segments(input_points, segments_n)

pcd.colors = o3d.utility.Vector3dVector(colors_l)
o3d.visualization.draw_geometries([pcd])

#%% Next Stages:
    
#1. Make it a function
#2. Treat Small Segments
#3. Compute segment-based features
#4. Refine the Segmentation with Hybrid Approach
#5. Unsupervised Approach (Parameter-less)
#6. Add to Production Line