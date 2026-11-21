
# Implementation of Expectation-Maximization Algorithm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Step 1: Generate Dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

print("Dataset Generated Successfully")
print("Total Samples:", len(X))

# Step 2: Initialize EM Algorithm
gmm = GaussianMixture(
    n_components=3,
    random_state=42,
    max_iter=100
)

# Step 3: Train the Model using EM
gmm.fit(X)

print("\nEM Algorithm Training Completed")

# Step 4: Predict Cluster Labels
labels = gmm.predict(X)

# Step 5: Display Results
print("\nCluster Labels:")
print(labels)

print("\nNumber of Clusters:", gmm.n_components)

print("Converged:", gmm.converged_)
print("Number of Iterations:", gmm.n_iter_)

# Step 6: Calculate Model Score
print("\nModel Score:", round(gmm.score(X), 4))

# Step 7: Plot Clustered Data
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis',
    s=40
)

plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Cluster Centers'
)

plt.title("Expectation-Maximization Algorithm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()

plt.show()

print("\nEM Algorithm Executed Successfully!")
