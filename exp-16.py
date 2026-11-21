
# Comparison of Different Classification Algorithms

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Step 3: Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Define classification algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# Step 5: Train and evaluate models
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average='weighted',
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average='weighted',
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average='weighted',
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

# Step 6: Display comparison
df = pd.DataFrame(results)

print("\nCLASSIFICATION ALGORITHM COMPARISON")
print(df.to_string(index=False))

# Step 7: Find best algorithm
best_model = df.loc[df['Accuracy'].idxmax()]

print("\nBEST CLASSIFICATION ALGORITHM")
print("Algorithm:", best_model['Algorithm'])
print("Accuracy:", best_model['Accuracy'], "%")

# Step 8: Plot accuracy comparison
plt.figure(figsize=(10, 5))

plt.bar(df['Algorithm'], df['Accuracy'])

plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Performance Comparison of Classification Algorithms")

plt.xticks(rotation=30)
plt.ylim(0, 110)
plt.tight_layout()
plt.show()
