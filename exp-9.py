
# Comparison of Linear and Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

print("Dataset Loaded Successfully")

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

# Step 4: Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)

poly_mse = mean_squared_error(y_test, poly_pred)
poly_r2 = r2_score(y_test, poly_pred)

# Step 5: Display Results
print("\nLINEAR REGRESSION RESULTS")
print("-------------------------")
print("MSE:", round(linear_mse, 2))
print("R2 Score:", round(linear_r2, 4))

print("\nPOLYNOMIAL REGRESSION RESULTS")
print("-----------------------------")
print("MSE:", round(poly_mse, 2))
print("R2 Score:", round(poly_r2, 4))

# Step 6: Compare Models
print("\nMODEL COMPARISON")

if poly_mse < linear_mse:
    print("Polynomial Regression performs better")
else:
    print("Linear Regression performs better")

# Step 7: Plot Graphs
X_curve = np.linspace(1, 10, 100).reshape(-1, 1)

linear_curve = linear_model.predict(X_curve)

poly_curve = poly_model.predict(
    poly.transform(X_curve)
)

plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X_curve, linear_curve,
         color='red', label='Linear Regression')

plt.plot(X_curve, poly_curve,
         color='green', label='Polynomial Regression')

plt.title("Linear vs Polynomial Regression")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.grid()

plt.show()

print("\nRegression Comparison Completed Successfully!")
