"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 14

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
#%% Step 2: Setting up our 3D python context

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

#for vizualisation purposes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%% Styling prompt

plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 600

#create paths and load data
data_folder="../DATA/"
result_folder="../RESULTS/"

#Load the file
dataset="3DML_urban_point_cloud.xyz"

#Store in a Pandas dataframe the content of the file on the google drive
pcd=pd.read_csv(data_folder+dataset,delimiter=' ')
pcd

#Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
pcd.dropna(inplace=True)

#%% Step 3: Point Cloud Feature selection and preparation

#Create training and testing
labels=pcd['Classification']
features=pcd[['X','Y','Z','R','G','B']]
features_scaled = MinMaxScaler().fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)

#this will take a lot of time
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])
clf.fit(X_train, y_train)

# No, scaling is not necessary for random forests. The nature of RF is such that convergence and numerical precision issues, which can sometimes trip up the algorithms used in logistic and linear regression, as well as neural networks, aren't so important
rf_classifier = RandomForestClassifier(n_estimators = 10)

#The line below is useful only if you want to create a classification model
rf_classifier.fit(X_train, y_train)

#The line below is useful only if you want to test on an unseen dataset (real scenario)
rf_predictions = rf_classifier.predict(X_test)

print(classification_report(y_test, rf_predictions, target_names=['ground','vegetation','buildings']))

# plotting the results 3D
ax = plt.axes(projection='3d')
ax.scatter(X_test['X'], X_test['Y'], X_test['Z'], c = rf_predictions, s=0.1)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].scatter(X_test['X'], X_test['Y'], c =y_test, s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(X_test['X'], X_test['Y'], c = rf_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(X_test['X'], X_test['Y'], c = y_test-rf_predictions, cmap = plt.cm.rainbow, s=0.5*(y_test-rf_predictions))
axs[2].set_title('Differences')

#%% Step 4: 3D Machine Learning Tuning

#Example of a K-Nearest Neighbors Model for 3D Point Cloud Semantic Segmentation
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print(classification_report(y_test, knn_predictions, target_names=['ground','vegetation','buildings']))

fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].scatter(X_test['X'], X_test['Y'], c =y_test, s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(X_test['X'], X_test['Y'], c = knn_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(X_test['X'], X_test['Y'], c = y_test-knn_predictions, cmap = plt.cm.rainbow, s=0.5*(y_test-knn_predictions))
axs[2].set_title('Differences')

#Example of a Multi-Layer Perception Model for 3D Point Cloud Semantic Segmentation
from sklearn.neural_network import MLPClassifier
mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 2), random_state=1)
mlp_classifier.fit(X_train, y_train)
mlp_predictions = mlp_classifier.predict(X_test)
print(classification_report(y_test, mlp_predictions, target_names=['ground','vegetation','buildings']))

fig, axs = plt.subplots(1, 3, figsize=(20,5)) # row 1, col 2 index 1
# fig, ax = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(X_test['X'], X_test['Y'], c =y_test, s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(X_test['X'], X_test['Y'], c = mlp_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(X_test['X'], X_test['Y'], c = y_test-mlp_predictions, cmap = plt.cm.rainbow, s=0.5*(y_test-mlp_predictions))
axs[2].set_title('Differences')

#%% Step 5: 3D Machine Learning Performance. Towards Generalization

val_dataset="3DML_validation.xyz"
val_pcd=pd.read_csv(data_folder+val_dataset,delimiter=' ')
val_pcd.dropna(inplace=True)

val_labels=val_pcd['Classification']
val_features=val_pcd[['X','Y','Z','R','G','B']]
val_predictions = rf_classifier.predict(val_features)
print(classification_report(val_labels, val_predictions, target_names=['ground','vegetation','buildings']))

fig, axs = plt.subplots(1, 3, figsize=(20,5)) # row 1, col 2 index 1
# fig, ax = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(val_features['X'], val_features['Y'], c =val_labels, s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(val_features['X'], val_features['Y'], c = val_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(val_features['X'], val_features['Y'], c = val_labels-val_predictions, cmap = plt.cm.rainbow, s=0.5*(val_labels-val_predictions))
axs[2].set_title('Differences')

labels=pcd['Classification']
features=pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']]
features_scaled = MinMaxScaler().fit_transform(features)

val_labels=val_pcd['Classification']
val_features=val_pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']]
val_features_scaled = MinMaxScaler().fit_transform(val_features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)
rf_classifier = RandomForestClassifier(n_estimators = 10)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print(classification_report(y_test, rf_predictions, target_names=['ground','vegetation','buildings']))

val_rf_predictions = rf_classifier.predict(val_features_scaled)
print(classification_report(val_labels, val_rf_predictions, target_names=['ground','vegetation','buildings']))

fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].scatter(val_pcd['X'], val_pcd['Y'], c =val_labels, s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(val_pcd['X'], val_pcd['Y'], c = val_rf_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(val_pcd['X'], val_pcd['Y'], c = val_labels-val_rf_predictions, cmap = plt.cm.rainbow, s=0.5*(val_labels-val_rf_predictions))
axs[2].set_title('Differences')

val_labels=val_pcd['Classification']
val_features=val_pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']]
val_features_sampled, val_features_test, val_labels_sampled, val_labels_test = train_test_split(val_features, val_labels, test_size=0.9)
val_features_scaled_sample = MinMaxScaler().fit_transform(val_features_test)

labels=pd.concat([pcd['Classification'],val_labels_sampled])
features=pd.concat([pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']],val_features_sampled])
features_scaled = MinMaxScaler().fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)
rf_classifier = RandomForestClassifier(n_estimators = 10)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

print(classification_report(y_test, rf_predictions, target_names=['ground','vegetation','buildings']))


val_rf_predictions_90 = rf_classifier.predict(val_features_scaled_sample)
print(classification_report(val_labels_test, val_rf_predictions_90, target_names=['ground','vegetation','buildings']))


val_features_scaled = MinMaxScaler().fit_transform(val_features)
val_rf_predictions = rf_classifier.predict(val_features_scaled)
print(classification_report(val_labels, val_rf_predictions, target_names=['ground','vegetation','buildings']))

# val_pcd['predictions']=val_rf_predictions
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].scatter(val_pcd['X'], val_pcd['Y'], c =val_pcd['Classification'], s=0.05)
axs[0].set_title('3D Point Cloud Ground Truth')
axs[1].scatter(val_pcd['X'], val_pcd['Y'], c = val_rf_predictions, s=0.05)
axs[1].set_title('3D Point Cloud Predictions')
axs[2].scatter(val_pcd['X'], val_pcd['Y'], c = val_pcd['Classification']-val_rf_predictions, cmap = plt.cm.rainbow, s=0.5*(val_pcd['Classification']-val_rf_predictions))
axs[2].set_title('Differences')
