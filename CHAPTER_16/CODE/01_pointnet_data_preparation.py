"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 16

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
* Status: Confidential

Dependencies:
* Anaconda or Miniconda
* An Anaconda new environment
* Libraries as described in the Chapter

Have fun with this Code Solution.

🎵 Note: Styling was not taken care of at this stage.

Enjoy!
"""

#%% 1. Python Set-up

#Base libraries
import numpy as np
import random
import torch
import torchnet as tnt

#Plotting libraries
import open3d as o3d

#Utilities libraries
from glob import glob 
import os
import functools

#%% 2. Data path setup

#specify data paths and extract filenames
project_dir = "../DATA/AERIAL_LOUHANS_COURSE"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

#%% 3. Train, Test and Validation Set Creation

valid_index = np.random.choice(len(pointcloud_train_files),int(len(pointcloud_train_files)/5), replace=False)
valid_list = [pointcloud_train_files[i] for i in valid_index]
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))),valid_index)]
test_list = pointcloud_test_files

print("%d tiles in train set, %d tiles in test set, %d files in valid list" % (len(train_list), len(test_list), len(valid_list)))

#%% 4. Quick Data Analysis

np.set_printoptions(precision=3)
tile_selected = pointcloud_train_files[random.randrange(20)]
print(tile_selected)

temp = np.loadtxt(tile_selected)

print('median\n',np.median(temp,axis=0))
print('std\n',np.std(temp,axis=0))
print('min\n',np.min(temp,axis=0))
print('max\n',np.max(temp,axis=0))

#%% 5. Computing the mean and the min of a data tile

cloud_data = temp.transpose()
min_f = np.min(cloud_data,axis=1)
mean_f = np.mean(cloud_data,axis=1)

print(min_f)
print(mean_f)

#%% 6. Normalize coordinates

n_coords = cloud_data[0:3]
n_coords[0] -= mean_f[0]
n_coords[1] -= mean_f[1]
n_coords[2] -= min_f[2]

print(n_coords)

#%% 7. Normalizing intensity

# The interquartile difference is the difference between the 75th and 25th quantile
IQR = np.quantile(cloud_data[-2],0.75)-np.quantile(cloud_data[-2],0.25)

# We subtract the median to all the observations and then dividing by the interquartile difference
n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)

#This permits to have a scaling robust to outliers (which is often the case)
n_intensity -= np.min(n_intensity)

print(n_intensity)

#%% 8. Definition of the cloud loading function

def cloud_loader(tile_name, features_used):
  cloud_data = np.loadtxt(tile_name).transpose()

  min_f=np.min(cloud_data,axis=1)
  mean_f=np.mean(cloud_data,axis=1)

  features=[]
  if 'xyz' in features_used:
    n_coords = cloud_data[0:3]
    n_coords[0] -= mean_f[0]
    n_coords[1] -= mean_f[1]
    n_coords[2] -= min_f[2]
    features.append(n_coords)
  if 'rgb' in features_used:
    colors = cloud_data[3:6]
    features.append(colors)
  if 'i' in features_used:
    IQR = np.quantile(cloud_data[-2],0.75)-np.quantile(cloud_data[-2],0.25)
    n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
    n_intensity -= np.min(n_intensity)
    features.append(n_intensity)
  
  gt = cloud_data[-1]
  gt = torch.from_numpy(gt).long()
  cloud_data = torch.from_numpy(np.vstack(features))
  return cloud_data, gt

#%% 9. Train, Test, Validation dataset Split

cloud_features = 'xyzrgbi'
test_set  = tnt.dataset.ListDataset(test_list,functools.partial(cloud_loader, features_used=cloud_features))
train_set = tnt.dataset.ListDataset(train_list,functools.partial(cloud_loader, features_used=cloud_features))
valid_set = tnt.dataset.ListDataset(valid_list,functools.partial(cloud_loader, features_used=cloud_features))



#%% 10. Point Cloud Tile  Vizualisation Function

def tile_vizualisation(tile_name, features_used='xyzrgbi'):
    cloud, gt = cloud_loader(tile_name, features_used)
    
    xyz = np.array(cloud[0:3]).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if 'rgb' in features_used:
        rgb = np.array(cloud[3:6]/255).transpose()
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    pcd.estimate_normals(fast_normal_computation=True)
    o3d.visualization.draw_geometries([pcd])    
    return

selection = valid_list[5]

tile_vizualisation(selection, features_used='xyzrgbi')