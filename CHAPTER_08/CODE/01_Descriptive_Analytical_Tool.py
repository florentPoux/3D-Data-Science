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
#%% 1. Importing libraries

#Base libraries
import numpy as np
import pandas as pd
import scipy as sp

#3D Library
import open3d as o3d

#Visualization and plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#%% 2. Styling prompt

plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 600

#Optional Configuration and package version
import sys
print(sys.version)
print(np.__version__,'\n', pd.__version__,'\n', sns.__version__,'\n', sp.__version__)

#%% 3. MetaData Exploration 

#Read File with Pandas
df_pcd = pd.read_csv('../DATA/verviers_features.csv', delimiter= ' ')

#Profiling the Dataset Quickly 
print('\n----------------------------------------------------\n',df_pcd.info(),
      '\n----------------------------------------------------\n',df_pcd.shape,
      '\n----------------------------------------------------\n',df_pcd.head(),
      '\n----------------------------------------------------\n',df_pcd.columns,
      '\n----------------------------------------------------\n',df_pcd.describe())

#Returns the number of unique values for each variable.
print('Unique values',df_pcd.nunique(axis=0))

# Get the column index names
feature_names = df_pcd.columns.tolist()



#%% 4. Geometry and Shape Analysis

xyz_o3d = o3d.geometry.PointCloud()
xyz_o3d.points = o3d.utility.Vector3dVector(np.array(df_pcd[['X','Y','Z']]))
xyz_o3d.colors = o3d.utility.Vector3dVector(np.array(df_pcd[['R','G','B']])/255)

#Number of points
p_pts = len(xyz_o3d.points)

o3d.visualization.draw_geometries([xyz_o3d])

#%% 5. Overall Dimensions
 
#Axis-Aligned Bounding-box
aabb = xyz_o3d.get_axis_aligned_bounding_box()
aabb.color = [1,0,0]

o3d.visualization.draw([xyz_o3d, aabb], bg_color = (1, 1, 1, 1), show_skybox = False, line_width = 5)

centroid = xyz_o3d.get_center()
aabb_center = aabb.get_center()

#%% 6. Volume Estimation (BB)
p_aabb_volume = aabb.volume()
print('Volume of AABB: ',p_aabb_volume)

# Better Volume Accuracy
oobb = xyz_o3d.get_minimal_oriented_bounding_box()
oobb.color = [0,1,0]
oobb.volume()
o3d.visualization.draw_geometries([xyz_o3d, oobb, aabb])

#Convex Hull
ch = xyz_o3d.compute_convex_hull()
# returns the triangle mesh, and the list of idxs of the points

o3d.visualization.draw_geometries([ch[0], xyz_o3d.select_by_index(ch[1])])
ch[0].get_volume()

#%% 7. Planarity Check

print(df_pcd['Planarity_(0.1)'].describe())
pla_as_colors = np.repeat(np.array(df_pcd[['Planarity_(0.1)']]), 3, axis=1)
xyz_o3d.colors = o3d.utility.Vector3dVector(pla_as_colors)
 
#changing the colormap
viridis_cmap = plt.colormaps['viridis']
viridis_array = viridis_cmap(pla_as_colors)
xyz_o3d.colors = o3d.utility.Vector3dVector(viridis_array[:,0,:3])
o3d.visualization.draw_geometries([xyz_o3d])

#%% 8. Curvature Analysis
print(df_pcd['Mean_curvature_(0.1)'].describe())

# apply the min-max scaling in Pandas using the .min() and .max() methods
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm
    
# call the min_max_scaling function
df_pcd_normalized = min_max_scaling(df_pcd)
df_pcd_normalized['Mean_curvature_(0.1)'].describe()

#Colorize based on the new normalized curvature
curv_as_colors = np.repeat(np.array(df_pcd[['Mean_curvature_(0.1)']]), 3, axis=1)
viridis_array = viridis_cmap(curv_as_colors)
xyz_o3d.colors = o3d.utility.Vector3dVector(viridis_array[:,0,:3])
o3d.visualization.draw_geometries([xyz_o3d])

#%% 9. Main Orientation
xyz_np = np.array(xyz_o3d.points)

def pca(point_cloud):
  # Center the point cloud
  centered_cloud = point_cloud - np.mean(point_cloud, axis=0)

  # Covariance matrix
  covariance = np.cov(centered_cloud.T)

  # Eigenvalues and eigenvectors
  eigenvalues, eigenvectors = np.linalg.eig(covariance)

  # Sort eigenvalues and eigenvectors in descending order
  sorted_index = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sorted_index]
  eigenvectors = eigenvectors[:, sorted_index]

  # Principal direction (eigenvector corresponding to largest eigenvalue)
  return eigenvalues, eigenvectors

# Principal direction
eig_val, eig_vec  = pca(xyz_np)
principal_dir = eig_vec[:, 0]
print(principal_dir)

#%% 10. Aspect Ratio

L, l, h = oobb.extent
ratio_Lh = L/h
ratio_lh = l/h
ratio_Ll = L/l


#%% 11. Single Object Case

meshes = ["../DATA/Car.ply", "../DATA/Soviet_Plane.ply"]

for data in meshes:
    mesh = o3d.io.read_triangle_mesh(data)
    mesh.compute_vertex_normals()
    #Axis-Aligne BB
    aabb = mesh.get_axis_aligned_bounding_box()
    aabb.color = [1,0,0]
    
    #Oriented BB
    oobb = mesh.get_minimal_oriented_bounding_box()
    oobb.color = [0,1,0]
    
    o3d.visualization.draw_geometries([mesh, oobb, aabb])
    
    L, l, h = oobb.extent
    ratio_Lh = L/h
    ratio_lh = l/h
    ratio_Ll = L/l
    
    print(ratio_Lh, ratio_lh, ratio_Ll)

#%% 12. Density Analysis

nn_distances = xyz_o3d.compute_nearest_neighbor_distance()
print(nn_distances)

#Central Tendency
mean_dist = round(np.mean(nn_distances),3)
print(f"Mean: {mean_dist} m")

median_dist = round(np.median(nn_distances),3)
print(f"Median: {median_dist} m")

#(2) Spread
std_dist = round(np.std(nn_distances),3)
print(f"Standard Deviation: {std_dist} m")

#(3) Percentiles
percentile25_dist = round(np.percentile(nn_distances, 25),3)
percentile50_dist = round(np.percentile(nn_distances, 50),3)
percentile75_dist = round(np.percentile(nn_distances, 75),3)
percentile95_dist = round(np.percentile(nn_distances, 95),3)
print(f"25% percentile: {percentile25_dist}")
print(f"50% percentile: {percentile50_dist}")
print(f"75% percentile: {percentile75_dist}")
print(f"95% percentile: {percentile95_dist}")


#(4) Skewness
skew_dist = sp.stats.skew(nn_distances)
mode_dist = sp.stats.mode(nn_distances)

print(f"Skew: {skew_dist}")
print(f"Mode: {mode_dist} m")

#%% 13. Histograms. Visualizing distributions

sel = feature_names[2]
print(f"{sel} statistics: \n {df_pcd[sel].describe()}")

#1
plt.hist(nn_distances)
plt.xlabel("Values (m)")
plt.ylabel("Frequency")
plt.title("Distribution of Neighbor Distances")
plt.show()

#2
plt.hist(df_pcd[sel])
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title(f"Distribution of {sel} feature")
plt.show()

#%% Styling the plot

def hist_styled(feature, f_name):
    # Create histogram
    n, bins, patches = plt.hist(feature, bins=100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=4)
    n = n.astype('int') # it MUST be integer
    
    # Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.cool(n[i]/max(n)))

    
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

hist_styled(nn_distances, 'NN_Distance')

#%% 3D Bar Chart
import matplotlib.colors as colplt
import matplotlib.cm as cm

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(df_pcd['X'], df_pcd['Y'], bins=100)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.1 * np.ones_like(zpos)
dz = hist.ravel()

offset = dz + np.abs(dz.min())
fracs = offset.astype(float)/offset.max()
norm = colplt.Normalize(fracs.min(), fracs.max())
colors = cm.cool(norm(fracs))

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.bar3d(xpos,ypos,zpos,0.1,0.1,dz, color=colors)
plt.show()



#%% Analysing patterns

# best fit of data
def hist_styled(feature, f_name, gaussian=True, f_stats=True):
    n, bins, patches = plt.hist(feature, bins=100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=3)
    n = n.astype('int')

    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.cool(n[i]/max(n)))
    
    if gaussian == True:    
        # Compute the Gaussian curve
        mu, sigma = sp.stats.norm.fit(feature)
        x = np.linspace(min(feature), max(feature), 100)
        gaussian = sp.stats.norm.pdf(x, mu, sigma)
        
        # Scale the Gaussian to match the histogram height
        scaling_factor = np.max(n) / np.max(gaussian)
        gaussian_scaled = gaussian * scaling_factor
    
        # Plot the Gaussian curve
        plt.plot(x, gaussian_scaled, color='deepskyblue', linewidth=2, label='Gaussian Fit', zorder=5)        

    if f_stats == True:
        # Calculate statistics
        mean_val = np.mean(feature)
        median_val = np.median(feature)
        mode_val = bins[np.argmax(n)]
        
        # Add vertical lines for mean, median, and mode
        plt.axvline(mean_val, color='lightcoral', linestyle='dashed', linewidth=1.3, label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='palegreen', linestyle='dashed', linewidth=1.3, label=f'Median: {median_val:.3f}')
        plt.axvline(mode_val, color='skyblue', linestyle='dashed', linewidth=1.3, label=f'Mode: {mode_val:.3f}')
        
        # Add legend
        plt.legend(fontsize=7)

    
    plt.title(r'$\mathrm{Histogram\ of\ %s:}\ \mu=%.3f,\ \sigma=%.3f$' %(sel, mu, sigma), fontsize=12)
    plt.xlabel(f'{f_name} values', fontsize=10)
    plt.ylabel('Point Numbers', fontsize=10)
    plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)
    plt.show()
    return

hist_styled(nn_distances, 'NN_Distance')

#%% Test other distributions Models

def hist_styled(feature, f_name, gaussian=True, distribs=False):
    n, bins, patches = plt.hist(feature, bins=100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7, zorder=3)
    n = n.astype('int') # it MUST be integer
    # Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.cool(n[i]/max(n)))
        
    if gaussian==True:
        # add a 'best fit' line
        mu, sigma = sp.stats.norm.fit(feature)
        p = sp.stats.norm.pdf(bins, mu, sigma)*plt.ylim()[1]
        plt.plot(bins, p, 'r--', linewidth=1, color = '#FE53BB', zorder=4)
        plt.title(r'$\mathrm{Histogram\ of\ %s:}\ \mu=%.3f,\ \sigma=%.3f$' %(sel, mu, sigma), fontsize=12)
    
    if distribs==True:
        dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
        for dist_name in dist_names:
            dist = getattr(sp.stats, dist_name)
            params = dist.fit(feature)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            if arg:
                pdf_fitted = dist.pdf(bins, *arg, loc=loc, scale=scale) * plt.ylim()[1]
            else:
                pdf_fitted = dist.pdf(bins, loc=loc, scale=scale) * plt.ylim()[1]
            plt.plot(bins, pdf_fitted, linewidth=1, label = dist_name, zorder=4)    
        plt.title('Multiple distributions', fontsize=12)
    
    plt.xlabel(f'{f_name} values', fontsize=10)
    plt.ylabel('Point Numbers', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)
    plt.show()
    return

hist_styled(df_pcd[sel], sel, gaussian=False, distribs=True)

#%% Density Estimate

sns.histplot(df_pcd[sel], kde=True, bins=100)

#%% 13. Box plot
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
]

plt.boxplot(df_pcd[sel], patch_artist=True, boxprops = dict(facecolor = colors[0], alpha = 0.7), medianprops = dict(color = colors[1], linewidth = 1), labels=[sel])
        
plt.ylabel("Value")
plt.title(f'{sel}-feature: BoxPlot', fontsize=12)
plt.grid(True, c = 'grey', ls = '--', lw = 0.1, zorder=0)
plt.show()

#%%
def boxplot_styled(feature, f_name):
    plt.boxplot(feature, patch_artist=True, boxprops = dict(facecolor = '#08F7FE', alpha = 0.7), medianprops = dict(color = '#FE53BB', linewidth = 1), showfliers = False, labels=[f_name])
            
    plt.ylabel("Value")
    plt.title(f'{f_name}-feature: BoxPlot', fontsize=10)
    plt.grid(True, c = 'grey', ls = '--', lw = 0.1, zorder=0)
    plt.show()
    return

boxplot_styled(df_pcd[sel], sel)
boxplot_styled(nn_distances, 'NN_Distance')

#%% Box plot XYZ
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
]

plt.boxplot(df_pcd[['X', 'Y', 'Z']], patch_artist=True, boxprops = dict(facecolor = colors[0], alpha = 0.7), medianprops = dict(color = colors[1], linewidth = 1.5), labels=['X', 'Y', 'Z'])
        
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.title("Boxplot of 3D coordinates")
plt.grid(True, c = 'grey', ls = '--', lw = 0.1, zorder=0)
plt.show()



#%% Box plot RGB
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
]

plt.boxplot(df_pcd[['R', 'G', 'B']], patch_artist=True, boxprops = dict(facecolor = colors[0], alpha = 0.7), medianprops = dict(color = colors[1], linewidth = 1.5), labels=['R', 'G', 'B'])
        
plt.xlabel("Colors")
plt.ylabel("Value")
plt.title("Boxplot of Colors")
plt.grid(True, c = 'grey', ls = '--', lw = 0.1, zorder=0)
plt.show()

#%% Correlations with Multiple variables

print(feature_names)
ei1 = '1st_eigenvalue_(0.1)'
ei2 = '2nd_eigenvalue_(0.1)'
ei3 = '3rd_eigenvalue_(0.1)'

#%% Density estimate multiple variables

sns.kdeplot(df_pcd[['X', 'Y', 'Z']], palette = 'cool', fill = True)
plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)

#%% RGB
sns.kdeplot(df_pcd[['R', 'G', 'B']], palette = 'cool', fill = True)
plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)

#%% Eigen Values
sns.kdeplot(df_pcd[[ei1, ei2, ei3]], palette = 'cool', fill = True)
plt.grid(True, c = 'grey', ls = '--', lw = 0.2, zorder=0)

#%% Correlation matrix

corr = df_pcd.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=False, cmap=sns.diverging_palette(220, 20, as_cmap=True))

#%% RGB Color

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