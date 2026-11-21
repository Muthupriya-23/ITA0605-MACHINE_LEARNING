
# Credit Score Classification using Random Forest

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Create Dataset
data = {
    'Income': [25000, 30000, 45000, 50000, 60000,
               70000, 80000, 90000, 35000, 55000,
               75000, 95000, 28000, 48000, 85000,
               65000, 40000, 100000, 32000, 72000],

    'Debt': [20000, 25000, 15000, 18000, 12000,
             10000, 8000, 5000, 28000, 15000,
             9000, 4000, 26000, 16000, 7000,
             11000, 22000, 3000, 24000, 9000],

    'Payment_History': [2, 2, 3, 3, 4,
                        4, 5, 5, 1, 3,
                        4, 5, 1, 3, 5,
                        4, 2, 5, 1, 4],

    'Credit_Utilization': [90, 85, 60, 55, 40,
                           30, 20, 15, 95, 50,
                           25, 10, 92, 58, 18,
                           35, 80, 8, 88, 28],

    'Credit_Score': [
        'Poor', 'Poor', 'Average', 'Average', 'Good',
        'Good', 'Good', 'Good', 'Poor', 'Average',
        'Good', 'Good', 'Poor', 'Average', 'Good',
        'Good', 'Poor', 'Good', 'Poor', 'Good'
    ]
}

df = pd.DataFrame(data)

print("Credit Score Dataset")
print(df.head())

# Step 2: Separate Features and Target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25,
    random_state=42, stratify=y
)

# Step 4: Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Train Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# Step 6: Predict Test Data
y_pred = model.predict(X_test)

# Step 7: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Credit Scores:")
print(y_pred)

print("\nActual Credit Scores:")
print(y_test.values)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 8: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, y_pred,
    labels=['Poor', 'Average', 'Good']
))

# Step 9: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    zero_division=0
))

# Step 10: Predict New Customer
new_customer = pd.DataFrame([{
    'Income': 60000,
    'Debt': 12000,
    'Payment_History': 4,
    'Credit_Utilization': 30
}])

prediction = model.predict(new_customer)

print("\nNew Customer Credit Score:")
print(prediction[0])

print("\nCredit Score Classification Completed Successfully!")
