
# Implementation of Linear Regression Algorithm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

print("Dataset Loaded Successfully")

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create Linear Regression Model
model = LinearRegression()

# Step 4: Train Model
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully")

# Step 5: Predict Output
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

# Step 6: Calculate Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Plot Regression Line
plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, model.predict(X), color='red',
         label='Regression Line')

plt.title("Linear Regression")
plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.legend()
plt.grid()

plt.show()

print("\nLinear Regression Completed Successfully!")
