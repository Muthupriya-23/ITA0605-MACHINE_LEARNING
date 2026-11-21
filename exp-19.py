
# Bank Loan Prediction using Naive Bayes Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Create sample bank loan dataset
data = {
    'Income': [25000, 30000, 35000, 40000, 45000,
               50000, 55000, 60000, 65000, 70000,
               75000, 80000, 85000, 90000, 95000,
               100000, 28000, 38000, 58000, 88000],

    'Credit_Score': [500, 520, 550, 580, 600,
                     620, 650, 680, 700, 720,
                     740, 760, 780, 800, 820,
                     850, 510, 570, 660, 790],

    'Loan_Amount': [200000, 250000, 300000, 250000,
                    300000, 350000, 300000, 400000,
                    350000, 450000, 400000, 500000,
                    450000, 550000, 500000, 600000,
                    250000, 300000, 350000, 500000],

    'Employment_Years': [1, 2, 2, 3, 3,
                         4, 5, 5, 6, 7,
                         7, 8, 9, 10, 11,
                         12, 1, 2, 5, 9],

    'Loan_Status': [0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

print("BANK LOAN DATASET")
print(df)

# Step 2: Separate input and output
X = df[['Income', 'Credit_Score',
        'Loan_Amount', 'Employment_Years']]

y = df['Loan_Status']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,
    stratify=y
)

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Evaluate model
print("\nMODEL ACCURACY")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred,
    target_names=['Rejected', 'Approved'],
    zero_division=0
))

# Step 8: Predict a new loan application
new_applicant = pd.DataFrame({
    'Income': [75000],
    'Credit_Score': [750],
    'Loan_Amount': [350000],
    'Employment_Years': [7]
})

prediction = model.predict(new_applicant)

print("\nNEW LOAN APPLICATION")
print(new_applicant)

if prediction[0] == 1:
    print("Loan Prediction: Approved")
else:
    print("Loan Prediction: Rejected")

# Step 9: Visualize loan status
df['Loan_Status'].value_counts().sort_index().plot(
    kind='bar'
)

plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.title("Bank Loan Approval Distribution")

plt.xticks(
    [0, 1],
    ['Rejected', 'Approved'],
    rotation=0
)

plt.tight_layout()
plt.show()
