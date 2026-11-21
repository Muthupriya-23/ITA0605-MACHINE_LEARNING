
# Perceptron Based Iris Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 2: Display dataset
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y

print("IRIS FLOWER DATASET")
print(df.head())

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Step 4: Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create Perceptron model
model = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42
)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predict test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\nMODEL ACCURACY")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Step 9: Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNEW FLOWER PREDICTION")
print("Predicted Species:",
      iris.target_names[prediction[0]])

# Step 10: Visualize Iris dataset
plt.scatter(
    iris.data[:, 0],
    iris.data[:, 1],
    c=iris.target,
    cmap='viridis'
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Perceptron Based Iris Classification")

plt.show()
