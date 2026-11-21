
# Implementation of Naive Bayes Algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# Step 3: Create Naive Bayes Classifier
model = GaussianNB()

# Step 4: Train the Model
model.fit(X_train, y_train)

print("\nNaive Bayes Model Trained Successfully")

# Step 5: Predict Test Data
y_pred = model.predict(X_test)

# Step 6: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 7: Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 8: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

print("\nNaive Bayes Classification Completed Successfully!")
