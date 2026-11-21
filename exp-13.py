
# Car Price Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create Car Dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019,
             2020, 2021, 2015, 2017, 2019,
             2020, 2021, 2016, 2018, 2022,
             2017, 2019, 2020, 2021, 2022],

    'Mileage': [50000, 45000, 40000, 35000, 30000,
                25000, 20000, 60000, 42000, 32000,
                28000, 18000, 55000, 38000, 15000,
                48000, 33000, 26000, 22000, 12000],

    'Engine': [1000, 1200, 1200, 1500, 1500,
               1800, 2000, 1000, 1200, 1500,
               1800, 2000, 1000, 1500, 2000,
               1200, 1500, 1800, 2000, 2200],

    'Horsepower': [70, 80, 85, 100, 110,
                   130, 150, 65, 90, 115,
                   135, 160, 75, 105, 170,
                   88, 120, 140, 155, 180],

    'Price': [300000, 350000, 420000, 500000, 600000,
              750000, 900000, 250000, 400000, 620000,
              780000, 950000, 320000, 520000, 1100000,
              390000, 650000, 820000, 920000, 1200000]
}

df = pd.DataFrame(data)

print("Car Dataset")
print(df.head())

# Step 2: Separate Features and Target
X = df[['Year', 'Mileage', 'Engine', 'Horsepower']]
y = df['Price']

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Linear Regression Model
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

print("\nCar Price Prediction Model Trained Successfully")

# Step 6: Predict Test Data
y_pred = model.predict(X_test)

print("\nActual Car Prices:")
print(y_test.values)

print("\nPredicted Car Prices:")
print(y_pred.round(2))

# Step 7: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# Step 8: Predict New Car Price
new_car = pd.DataFrame([{
    'Year': 2021,
    'Mileage': 20000,
    'Engine': 1500,
    'Horsepower': 120
}])

predicted_price = model.predict(new_car)

print("\nNew Car Details:")
print(new_car)

print("\nEstimated Car Price: Rs.",
      round(predicted_price[0], 2))

# Step 9: Plot Actual vs Predicted Prices
plt.scatter(y_test, y_pred, color='blue')

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.title("Car Price Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid()

plt.show()

print("\nCar Price Prediction Completed Successfully!")
