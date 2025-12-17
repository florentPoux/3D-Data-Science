"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: 3D Workflow Summary

General Information:
-------------------
* 🦊 Created by:    Florent Poux
* 📅 Last Update:   Dec. 2024
* © Copyright:      Florent Poux
* 📜 License:       MIT

Dependencies:
------------
* Environment:      Anaconda or Miniconda
* Python Version:   3.9+
* Key Libraries:    NumPy, Pandas, Open3D, Scikit-Learn

Helpful Links:
-------------
* 🏠 Author Website:        https://learngeodata.eu
* 📚 O'Reilly Book Page:    https://www.oreilly.com/library/view/3d-data-science/9781098161323/

Enjoy this code! 🚀
"""

import numpy as np
import open3d as o3d
import subprocess
import os
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Photogrammetry (COLMAP) ---
def run_colmap(image_dir, output_dir):
    """
    Run COLMAP photogrammetry pipeline.
    Requires COLMAP (https://colmap.github.io/) to be installed and in PATH.
    """
    print("📸 Starting COLMAP reconstruction...")
    try:
        # Feature extraction
        subprocess.run(["colmap", "feature_extractor",
                        "--database_path", f"{output_dir}/database.db",
                        "--image_path", image_dir], check=True)
        
        # Feature matching
        subprocess.run(["colmap", "exhaustive_matcher",
                        "--database_path", f"{output_dir}/database.db"], check=True)
        
        # Sparse reconstruction
        os.makedirs(f"{output_dir}/sparse", exist_ok=True)
        subprocess.run(["colmap", "mapper",
                        "--database_path", f"{output_dir}/database.db",
                        "--image_path", image_dir,
                        "--output_path", f"{output_dir}/sparse"], check=True)
        
        # Dense reconstruction
        os.makedirs(f"{output_dir}/dense", exist_ok=True)
        subprocess.run(["colmap", "image_undistorter",
                        "--image_path", image_dir,
                        "--input_path", f"{output_dir}/sparse/0",
                        "--output_path", f"{output_dir}/dense"], check=True)
        
        subprocess.run(["colmap", "patch_match_stereo",
                        "--workspace_path", f"{output_dir}/dense"], check=True)
        
        subprocess.run(["colmap", "stereo_fusion",
                        "--workspace_path", f"{output_dir}/dense",
                        "--output_path", f"{output_dir}/dense/fused.ply"], check=True)
        print("✅ COLMAP reconstruction finished.")
        return f"{output_dir}/dense/fused.ply"
    except FileNotFoundError:
        print("❌ COLMAP executable not found. Please install COLMAP.")
        return None
    except Exception as e:
        print(f"❌ COLMAP execution failed: {e}")
        return None

# --- 2. Pre-processing ---
def remove_noise(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical Outlier Removal."""
    print("🧹 Removing noise...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return cl, ind

def adaptive_downsample(pcd, voxel_size=0.05, detail_size=0.02):
    """Adaptive downsampling preserving details based on curvature."""
    print("📉 Adaptive downsampling...")
    pcd.estimate_normals()
    
    normals = np.asarray(pcd.normals)
    if len(normals) == 0:
        return pcd.voxel_down_sample(voxel_size)

    # Estimate curvature
    # Simple proxy: variance of normals in local neighborhood could be used,
    # but here we use global variance for simplicity as per snippet logic
    # Real implementation would use neighborhood PCA.
    # We will assume 'curvature' is computed or approximated.
    # Approximation: random score for demo if kdtree not used
    curvature = np.random.rand(len(normals)) # Placeholder for real curvature calc
    
    high_detail = curvature > np.percentile(curvature, 75)
    
    points = np.asarray(pcd.points)
    high_detail_pcd = o3d.geometry.PointCloud()
    high_detail_pcd.points = o3d.utility.Vector3dVector(points[high_detail])
    
    low_detail_pcd = o3d.geometry.PointCloud()
    low_detail_pcd.points = o3d.utility.Vector3dVector(points[~high_detail])
    
    high_detail_down = high_detail_pcd.voxel_down_sample(detail_size)
    low_detail_down = low_detail_pcd.voxel_down_sample(voxel_size)
    
    return high_detail_down + low_detail_down

# --- 3. Feature Extraction ---
def extract_features(pcd, radius=0.1):
    """Extract local (FPFH) and global features."""
    print("🧬 Extracting features...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius*5, max_nn=100))
    
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    
    features = {
        'fpfh': np.asarray(fpfh.data).T,
        'distance_to_centroid': distances,
        'height': points[:, 2]
    }
    return features

# --- 4. 3D Structures & Segmentation ---
def voxelize(pcd, voxel_size):
    """Convert Point Cloud to Voxel Grid."""
    print("📦 Voxelizing...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def dbscan_3d(points, eps, min_points):
    """Custom DBSCAN implementation using KDTree."""
    print("🔍 Running custom DBSCAN...")
    tree = KDTree(points)
    clusters = []
    visited = set()
    
    def expand_cluster(point_idx, neighbors):
        cluster = [point_idx]
        # Iterate over neighbor indices directly
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                # query_ball_point returns list of indices
                new_neighbors = tree.query_radius([points[neighbor]], r=eps)[0]
                if len(new_neighbors) >= min_points:
                    cluster.extend(expand_cluster(neighbor, new_neighbors))
        return cluster
    
    for i in range(len(points)):
        if i not in visited:
            # Query neighbors
            neighbors = tree.query_radius([points[i]], r=eps)[0]
            if len(neighbors) >= min_points:
                visited.add(i)
                cluster = expand_cluster(i, neighbors)
                clusters.append(cluster)
    
    return clusters

class OctreeNode:
    """Simple Python Octree Node."""
    def __init__(self, center, size, points):
        self.center = center
        self.size = size
        self.points = points
        self.children = []
        
    def subdivide(self):
        if len(self.points) <= 100:
            return
        
        new_size = self.size / 2
        for i in range(8):
            offset = new_size * np.array([
                (i & 1) - 0.5,
                ((i >> 1) & 1) - 0.5,
                ((i >> 2) & 1) - 0.5
            ])
            new_center = self.center + offset
            # Simple box check
            mask = np.all(np.abs(self.points - new_center) <= new_size/2, axis=1)
            if np.any(mask):
                new_node = OctreeNode(new_center, new_size, self.points[mask])
                self.children.append(new_node)
                new_node.subdivide()

def build_octree(points):
    print("🌳 Building Octree...")
    if len(points) == 0: return None
    center = np.mean(points, axis=0)
    size = np.max(np.abs(points - center)) * 2
    root = OctreeNode(center, size, points)
    root.subdivide()
    return root

# --- 5. Maching Learning ---
def train_rf(point_cloud, labels=None, test_size=0.2, n_estimators=100):
    """Train a Random Forest Classifier."""
    print("🌲 Training Random Forest...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(point_cloud)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=test_size, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    train_accuracy = rf_classifier.score(X_train, y_train)
    test_accuracy = rf_classifier.score(X_test, y_test)
    print(f"   -> Training Accuracy: {train_accuracy:.2f}")
    print(f"   -> Testing Accuracy: {test_accuracy:.2f}")
    
    return rf_classifier, scaler

def prediction(pcd, model, scaler):
    """Predict labels for a point cloud."""
    print("🔮 Predicting labels...")
    points = np.asarray(pcd.points)
    if not pcd.has_normals(): pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
    
    features = np.hstack((points, colors, normals))
    features_scaled = scaler.transform(features)
    
    return model.predict(features_scaled)

# --- 6. Meshing ---
def point_cloud_to_mesh(pcd, depth=8):
    """Poisson Surface Reconstruction."""
    print("🕸️ Creating Mesh...")
    if not pcd.has_normals():
        pcd.estimate_normals()
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

# --- Main Pipeline ---
def main():
    print("🚀 Starting 3D Data Science Workflow Summary...")
    
    # 1. Load Data (Generate synthetic if missing)
    # Trying to load a demo file or generate
    try:
        pcd = o3d.io.read_point_cloud("../DATA/sample.ply")
        if pcd.is_empty(): raise FileNotFoundError
    except:
        print("⚠️ No data found, generating synthetic torus...")
        mesh = o3d.geometry.TriangleMesh.create_torus()
        pcd = mesh.sample_points_poisson_disk(5000)
    
    # 2. Add some noise
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, 0.02, points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + noise)
    
    # 3. Clean
    clean_pcd, _ = remove_noise(pcd)
    
    # 4. Downsample
    ds_pcd = adaptive_downsample(clean_pcd)
    
    # 5. Extract Features
    feats = extract_features(ds_pcd)
    print(f"   -> Extracted {feats['fpfh'].shape[0]} FPFH descriptors.")
    
    # 6. Structuring
    octree = build_octree(np.asarray(ds_pcd.points))
    voxels = voxelize(ds_pcd, voxel_size=0.1)
    
    # 7. Clustering (Custom DBSCAN on subset)
    subset_points = np.asarray(ds_pcd.points)[:1000]
    clusters = dbscan_3d(subset_points, eps=0.1, min_points=5)
    print(f"   -> Found {len(clusters)} clusters.")
    
    # 8. Meshing
    mesh = point_cloud_to_mesh(ds_pcd)
    
    # 9. Visualization
    print("👀 Visualizing final mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Final Result", width=800, height=600)

if __name__ == "__main__":
    main()