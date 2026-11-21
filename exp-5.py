
# Implementation of K-Nearest Neighbours (K-NN)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Step 3: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Create K-NN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Step 5: Train the Model
knn.fit(X_train, y_train)

print("\nK-NN Model Trained Successfully")

# Step 6: Predict Test Data
y_pred = knn.predict(X_test)

# Step 7: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 8: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Step 10: Predict a New Sample
sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = knn.predict(sample_scaled)

print("\nNew Sample Prediction:")
print("Predicted Class:", iris.target_names[prediction[0]])

print("\nK-NN Classification Completed Successfully!")
