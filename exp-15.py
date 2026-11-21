
# Iris Flower Classification using Naive Bayes

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Naive Bayes classifier
model = GaussianNB()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("\nMODEL ACCURACY")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Step 8: Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNEW FLOWER PREDICTION")
print("Predicted Species:",
      iris.target_names[prediction[0]])

# Step 9: Visualize classification
plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap='viridis'
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Flower Classification using Naive Bayes")
plt.show()
