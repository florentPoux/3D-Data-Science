"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 12

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

#%% 1 Importing the library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


from sklearn.decomposition import PCA

#%% 2. Selecting a scene
data_folder="../DATA/"
dataset="KME_planes.xyz"

x,y,z,illuminance,reflectance,intensity,nb_of_returns = np.loadtxt(data_folder+dataset,skiprows=1, delimiter=';', unpack=True)

#%% 3. Isolating a specifc part that we want to decompose in a number of clusters

#Vizualising on the two axes to select the filter
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.scatter(x, z, c=intensity, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("First view")
plt.xlabel('X-axis ')
plt.ylabel('Z-axis ')

plt.subplot(1, 2, 2) # index 2
plt.scatter(y, z, c=intensity, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("Second view")
plt.xlabel('Y-axis ')
plt.ylabel('Z-axis ')

plt.show()

#%% 4. filtering points based on our threshold
pcd=np.column_stack((x,y,z))
idx=np.where(z>np.mean(z))
mask=z>np.mean(z)
spatial_query=pcd[z>np.mean(z)]
print(pcd.shape, spatial_query.shape)

#%% 5. Visualization

#plotting the results 3D
ax = plt.axes(projection='3d')
ax.scatter(x[mask], y[mask], z[mask], c = intensity[mask], s=0.1)
plt.show()

#plotting the results 2D
plt.scatter(x[mask], y[mask], c=intensity[mask], s=0.1)
plt.show()

#running the inference on the spatial coordinates
X=np.column_stack((x[mask], y[mask]))
kmeans = KMeans(n_clusters=20, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

X=np.column_stack((x[mask], y[mask], z[mask], illuminance[mask], nb_of_returns[mask], intensity[mask]))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

X=np.column_stack((z[mask] ,z[mask], intensity[mask]))
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

#%% 6. K-Means on the data

X=np.column_stack((z,nb_of_returns,intensity))
kmeans = KMeans(n_clusters=3).fit(X)

#plot 3D
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c = kmeans.labels_, s=20)
plt.show()

#plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'red')
plt.scatter(x, y, c='blue', s=20)
plt.show()

#values
# array([[290.41736598, 192.3429344 ,   6.64448367],
#        [287.37291391, 194.47415009,   6.66347143],
#        [290.89690944, 194.42305795,   6.63686364]])

#%% Optional. Exporting the dataset
result_folder="../DATA/RESULTS/"
np.savetxt(result_folder+dataset.split(".")[0]+"_result.xyz", np.column_stack((x[mask], y[mask], z[mask],kmeans.labels_)), fmt='%1.4f',  delimiter=';')

#%% 7. Run on cars

data_folder="../DATA/"
dataset="KME_cars.xyz"
x,y,z,r,g,b = np.loadtxt(data_folder+dataset,skiprows=1, delimiter=';', unpack=True)
X=np.column_stack((x,y,z))
kmeans = KMeans(n_clusters=3).fit(X)

#%% 8. analysis on dbscan
clustering = DBSCAN(eps=0.4, min_samples=2).fit(X)
plt.scatter(x, y, c=clustering.labels_, s=20)
plt.show()

#%% 9. Other experiments
X=np.column_stack((x,y,z))
pca = PCA(n_components=3)
Y=pca.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0).fit(Y)

ax = plt.axes(projection='3d')
ax.scatter(Y[:,0], Y[:,1], Y[:,2], c = kmeans.labels_, s=0.01)
plt.show()

#%% 10. Playing with feature spaces

X=np.column_stack((x[mask], y[mask], z[mask]))
wcss = [] 
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 20), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()
