
# Future Sales Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create sample sales dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Sales': [100, 120, 130, 150, 160, 180,
              190, 210, 220, 240, 250, 270]
}

df = pd.DataFrame(data)

print("SALES DATASET")
print(df)

# Step 2: Separate input and output
X = df[['Month']]
y = df['Sales']

# Step 3: Create Linear Regression model
model = LinearRegression()

# Step 4: Train the model
model.fit(X, y)

# Step 5: Predict existing sales
y_pred = model.predict(X)

# Step 6: Evaluate the model
print("\nMODEL EVALUATION")
print("Mean Absolute Error:",
      round(mean_absolute_error(y, y_pred), 2))

print("Mean Squared Error:",
      round(mean_squared_error(y, y_pred), 2))

print("R2 Score:",
      round(r2_score(y, y_pred), 2))

# Step 7: Predict future sales
future_months = pd.DataFrame({
    'Month': [13, 14, 15, 16, 17, 18]
})

future_sales = model.predict(future_months)

print("\nFUTURE SALES PREDICTION")

for month, sale in zip(future_months['Month'], future_sales):
    print("Month", month, "Predicted Sales:",
          round(sale, 2))

# Step 8: Visualize actual and predicted sales
plt.scatter(X['Month'], y, label="Actual Sales")

plt.plot(X['Month'], y_pred,
         color='red', label="Predicted Sales")

plt.plot(future_months['Month'], future_sales,
         marker='o', linestyle='--',
         label="Future Sales")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction using Linear Regression")

plt.legend()
plt.grid(True)
plt.show()
