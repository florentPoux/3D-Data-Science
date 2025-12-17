"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 15

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

#%% Libraries import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import laspy
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import numpy as np
from sklearn.model_selection import train_test_split

#%%

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%%

class PointCloudCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(PointCloudCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Example usage
num_classes = 10  # Adjust based on your dataset
model = PointCloudCNN(num_classes)

# Example input (batch_size, channels, depth, height, width)
example_input = torch.randn(1, 1, 64, 64, 64)
output = model(example_input)
print(output.shape)  # Should be (1, num_classes)
    
#%% Definition of training and validation

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=2)
        correct += (pred == target).sum().item()
        total += target.numel()
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
            pred = output.argmax(dim=2)
            correct += (pred == target).sum().item()
            total += target.numel()
    return total_loss / len(val_loader), correct / total

#%% Definition of Class

class LasPointCloudDataset(Dataset):
    def __init__(self, las_file_path, num_points=4096, voxel_size=1.0, features=['intensity', 'return_number', 'number_of_returns']):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.features = features

        # Read the .las file
        las = laspy.read(las_file_path)

        # Extract point cloud data
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Extract features
        feature_data = []
        for feature in features:
            if hasattr(las, feature):
                feature_data.append(getattr(las, feature))
            else:
                print(f"Warning: Feature '{feature}' not found in the .las file.")
        
        feature_data = np.vstack(feature_data).transpose() if feature_data else np.zeros((len(points), 0))
        
        # Combine points and features
        self.point_cloud = np.hstack((points, feature_data))
        self.labels = las.classification

        # Normalize spatial coordinates
        self.scaler = StandardScaler()
        self.point_cloud[:, :3] = self.scaler.fit_transform(self.point_cloud[:, :3])

        # Create a KD-tree for efficient nearest neighbor search
        self.kdtree = cKDTree(self.point_cloud[:, :3])
        
        # Get unique classes
        self.unique_classes = np.unique(self.labels)
        self.num_classes = len(self.unique_classes)

    def __len__(self):
        return len(self.point_cloud) // self.num_points

    def __getitem__(self, idx):
        # Randomly select a center point
        center_idx = np.random.randint(0, len(self.point_cloud))
        center = self.point_cloud[center_idx]

        # Find nearest neighbors
        _, indices = self.kdtree.query(center, k=self.num_points)
        sample = self.point_cloud[indices]
        sample_labels = self.labels[indices]

        # Voxelize the sample
        voxelized, voxel_labels = self.voxelize(sample, sample_labels)

        # Convert to tensor
        voxelized = torch.FloatTensor(voxelized)
        voxel_labels = torch.LongTensor(voxel_labels)

        return voxelized, voxel_labels

    def voxelize(self, points, labels):
        # Determine the number of voxels in each dimension
        voxel_dims = np.ceil((points.max(axis=0) - points.min(axis=0)) / self.voxel_size).astype(int)
        
        # Initialize voxel grid and label grid
        voxel_grid = np.zeros((1, *voxel_dims))  # Only one channel for point count
        label_grid = np.zeros(voxel_dims, dtype=int)

        # Voxelize points and labels
        for point, label in zip(points, labels):
            voxel_index = tuple((point - points.min(axis=0)) // self.voxel_size)
            voxel_grid[0][voxel_index] += 1  # Count of points in voxel
            label_grid[voxel_index] = label  # Assign label to voxel (last label wins)

        return voxel_grid, label_grid

def create_dataloader(las_file_path, batch_size=32, num_points=4096, voxel_size=1.0, features=['intensity', 'return_number', 'number_of_returns']):
    dataset = LasPointCloudDataset(las_file_path, num_points, voxel_size, features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#%% Execution

# Assuming you have a dataset of point clouds and labels
# Each point cloud is a numpy array of shape (num_points, 3)
# Labels are numpy arrays of shape (num_points,)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the features to use
features = []
# features = ['intensity', 'return_number', 'number_of_returns']

# Load and split the data
las_file_path = 'DATA/indoor_test.las'
full_dataset = LasPointCloudDataset(las_file_path, num_points=4096, voxel_size=1.0, features=features)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

# Initialize the model
num_classes = full_dataset.num_classes
input_channels = len(features) + 1  # +1 for the point count in each voxel
model = PointCloudCNN(num_classes, input_channels).to(device)

# Choose loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")

# Test the model
test_loss, test_accuracy = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#%% Prediction and Export

def predict_and_save(model, test_loader, original_las_path, output_las_path, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            predictions = output.argmax(dim=2).cpu().numpy()
            all_predictions.extend(predictions.reshape(-1))

    # Read the original .las file
    with laspy.open(original_las_path) as original_las:
        # Create a new .las file with the same properties as the original
        with laspy.open(output_las_path, mode="w", header=original_las.header) as output_las:
            # Copy all points from the original file to the new file
            output_las.write_points(original_las.read_points())

            # Add a new field for predictions
            output_las.add_extra_dim(laspy.ExtraBytesParams(name="prediction", type=np.int32))

            # Write predictions to the new field
            output_las.prediction = np.array(all_predictions[:len(output_las.points)])

    print(f"Predictions saved to {output_las_path}")