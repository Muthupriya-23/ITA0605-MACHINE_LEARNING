
# Implementation of Logistic Regression Algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load Dataset
iris = load_iris()

# Step 2: Select Two Classes
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

print("Iris Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Step 4: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create Logistic Regression Model
model = LogisticRegression()

# Step 6: Train Model
model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully")

# Step 7: Predict Test Data
y_pred = model.predict(X_test)

# Step 8: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 9: Display Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names[:2]
))

print("\nLogistic Regression Completed Successfully!")
