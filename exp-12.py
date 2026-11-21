
# Iris Flower Classification using KNN

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Step 5: Train the Model
knn.fit(X_train, y_train)

print("\nKNN Model Trained Successfully")

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
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 9: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Step 10: Predict a New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = knn.predict(new_flower_scaled)

print("\nNew Flower Prediction:")
print("Flower Species:", iris.target_names[prediction[0]])

# Step 11: Display Graph
plt.figure(figsize=(7, 5))

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap='viridis',
    s=60
)

plt.title("Iris Flower Classification using KNN")
plt.xlabel("Sepal Length (Scaled)")
plt.ylabel("Sepal Width (Scaled)")
plt.grid()

plt.show()

print("\nIris Flower Classification Completed Successfully!")
