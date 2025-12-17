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
3D Point Cloud PCA

Created by Florent Poux, (c) Licence MIT
To reuse in your project, please cite the most appropriate article accessible on my Google Scholar page

Have fun with this script!
"""

#%% 1. Importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.dpi'] = 600 

#%% 2. Load a point cloud .xyz from the previous step with Pandas
data_folder="../DATA/"
dataset="velodyne_pca.xyz"
pcd = pd.read_csv(data_folder+dataset, delimiter=";", names=['X', 'Y', 'Z','LABEL'] , header=None)

pcd['LABEL'] = pcd['LABEL'].astype(int)

#%% 3. Prepare Clusters as single objects: Split Dataframe into groups

segments=pcd.groupby(['LABEL'])
# segments.get_group(2) to get the group values
# segments.groups[2] to get the indexes of the df



#%% 4. Apply some operations to each of those smaller tables: PCA
cluster=segments.get_group(3)[['X','Y','Z']]

#compute the mean and center it
m = np.mean(cluster, axis=0)
cluster_norm = cluster-m

cov = np.cov(cluster_norm.T)
eig_val, eig_vec = np.linalg.eig(cov.T)

sorted_indexes = np.argsort(eig_val)[::-1]
eig_val = eig_val[sorted_indexes]
eig_vec = eig_vec[:,sorted_indexes]

#%% 5. Preparation of a 3D Plot Scene

fig= plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([min(cluster['X'])-1,max(cluster['X']+1)])
ax.set_ylim([min(cluster['Y'])-1,max(cluster['Y']+1)])
ax.set_zlim([min(cluster['Z'])-1,max(cluster['Z']+1)])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_title('PCA of Cluster'+' 2',fontsize = 10)


#%% 6. Qualitative Analysis: Plot in 3D the results of the PCA above
u = eig_vec[0]
v = eig_vec[1]
w = eig_vec[2]

ax.scatter(cluster['X'], cluster['Y'], cluster['Z'], color='steelblue', alpha=0.2)
ax.quiver(m[0],m[1],m[2],u[0],u[1],u[2], color ='salmon')
ax.quiver(m[0],m[1],m[2],v[0],v[1],v[2], color ='royalblue')
ax.quiver(m[0],m[1],m[2],w[0],w[1],w[2], color ='darkturquoise')



#%% 7. Define two functions: DrawPCA and CustomPCA

def DrawPCA(points,id_cluster):
    m = np.mean(points, axis=0)
    cluster_norm = points-m
    
    cov = np.cov(cluster_norm.T)
    eig_val, eig_vec = np.linalg.eig(cov.T)
    
    sorted_indexes = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indexes]
    eig_vec = eig_vec[:,sorted_indexes]
    
    u = eig_vec[0]
    v = eig_vec[1]
    w = eig_vec[2]
    
    fig= plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([min(cluster['X'])-1,max(cluster['X']+1)])
    ax.set_ylim([min(cluster['Y'])-1,max(cluster['Y']+1)])
    ax.set_zlim([min(cluster['Z'])-1,max(cluster['Z']+1)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_title('PCA of Cluster '+str(id_cluster),fontsize = 10)
    
    
    ax.scatter(cluster['X'], cluster['Y'], cluster['Z'], color='steelblue', alpha=0.2)
    ax.quiver(m[0],m[1],m[2],u[0],u[1],u[2], color ='darkturquoise')
    ax.quiver(m[0],m[1],m[2],v[0],v[1],v[2], color ='royalblue')
    ax.quiver(m[0],m[1],m[2],w[0],w[1],w[2], color ='salmon')    
    return 

def CustomPCA(points):
    m = np.mean(points, axis=0)
    cluster_norm = points-m
    
    cov = np.cov(cluster_norm.T)
    eig_val, eig_vec = np.linalg.eig(cov.T)
    
    sorted_indexes = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indexes]
    eig_vec = eig_vec[:,sorted_indexes]
    return eig_val, eig_vec

#%% 8. Loop over n segments to export images with PCA vectors
image_folder="../IMAGES/"
for i in range(10):
    cluster=segments.get_group(i)[['X','Y','Z']]
    DrawPCA(cluster,i)
    plt.savefig(image_folder+'PCA of Cluster '+str(i)+'.png', dpi=600)

#%% 9. Define a feature extraction loop for eigen values and normals
pcd['eig_1'],pcd['eig_2'],pcd['eig_3']=0,0,0
pcd['nx'],pcd['ny'],pcd['nz']=0,0,0

t1= time.time()    
for i in range(len(segments)):   
    cluster=segments.get_group(i)[['X','Y','Z']]
    eig_val, eig_vec = CustomPCA(cluster) 
    pcd.loc[cluster.index.values,['eig_1','eig_2','eig_3']] = eig_val[0],eig_val[1],eig_val[2]
    pcd.loc[cluster.index.values,['nx','ny','nz']] = eig_vec[2][0],eig_vec[2][1],eig_vec[2][2]
t2=time.time()
print("Time to attribute features to segments ", t2-t1, " seconds")

#%% 10. Exporting the data with the Eigen features
result_folder="../DATA/RESULTS/"
# pcd.to_csv(result_folder+dataset.split(".")[0]+"_PCA.xyz",float_format='%1.9f',index=False)