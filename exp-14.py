
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create sample house dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3200,
             3500, 3800, 4000, 4200, 4500],

    'Bedrooms': [2, 2, 3, 3, 3,
                 3, 4, 4, 4, 4,
                 4, 5, 5, 5, 5],

    'Bathrooms': [1, 2, 2, 2, 3,
                  3, 3, 3, 3, 4,
                  4, 4, 4, 5, 5],

    'Age': [15, 12, 10, 8, 7,
            6, 5, 4, 3, 5,
            4, 3, 2, 2, 1],

    'Price': [25, 30, 38, 42, 48,
              52, 60, 65, 70, 75,
              82, 88, 95, 100, 110]
}

df = pd.DataFrame(data)

print("HOUSE PRICE DATASET")
print(df)

# Step 2: Separate input and output
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("\nMODEL EVALUATION")
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, y_pred), 2))

print("Mean Squared Error:",
      round(mean_squared_error(y_test, y_pred), 2))

print("R2 Score:",
      round(r2_score(y_test, y_pred), 2))

# Step 7: Predict price of a new house
new_house = pd.DataFrame({
    'Area': [2500],
    'Bedrooms': [3],
    'Bathrooms': [3],
    'Age': [5]
})

predicted_price = model.predict(new_house)

print("\nNEW HOUSE DETAILS")
print(new_house)

print("Predicted House Price: Rs.",
      round(predicted_price[0] * 100000, 2))

# Step 8: Visualize actual vs predicted prices
plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price (Lakhs)")
plt.ylabel("Predicted House Price (Lakhs)")
plt.title("House Price Prediction")

plt.show()
