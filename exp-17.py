
# Mobile Price Prediction using Random Forest Classifier

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Create mobile phone dataset
data = {
    'RAM': [2, 2, 3, 3, 4, 4, 4, 6, 6, 6,
            8, 8, 8, 12, 12, 16],

    'Storage': [16, 32, 32, 64, 64, 128, 64, 128,
                128, 256, 128, 256, 256, 256, 512, 512],

    'Battery': [2500, 3000, 3000, 3500, 3500, 4000,
                4000, 4000, 4500, 4500, 4500, 5000,
                5000, 5000, 5500, 6000],

    'Camera': [8, 8, 12, 12, 13, 13, 16, 16,
               20, 20, 32, 32, 48, 48, 64, 108],

    'Price_Range': [0, 0, 0, 1, 1, 1, 1, 2,
                    2, 2, 2, 3, 3, 3, 3, 3]
}

df = pd.DataFrame(data)

print("MOBILE PHONE DATASET")
print(df)

# Step 2: Separate input and output
X = df[['RAM', 'Storage', 'Battery', 'Camera']]
y = df['Price_Range']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,
    stratify=y
)

# Step 4: Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

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
    y_test, y_pred, zero_division=0
))

# Step 8: Predict price range of a new mobile
new_mobile = pd.DataFrame({
    'RAM': [8],
    'Storage': [128],
    'Battery': [4500],
    'Camera': [48]
})

prediction = model.predict(new_mobile)

price_categories = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("\nNEW MOBILE DETAILS")
print(new_mobile)

print("Predicted Mobile Price Range:",
      price_categories[prediction[0]])

# Step 9: Visualize price range distribution
df['Price_Range'].value_counts().sort_index().plot(
    kind='bar'
)

plt.xlabel("Price Range")
plt.ylabel("Number of Mobiles")
plt.title("Mobile Price Range Distribution")
plt.xticks(
    [0, 1, 2, 3],
    ["Low", "Medium", "High", "Very High"],
    rotation=0
)

plt.tight_layout()
plt.show()
