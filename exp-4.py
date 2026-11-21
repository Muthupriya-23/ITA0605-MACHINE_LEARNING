
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset: Iris")
print("Total Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# 2. Convert output into one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2,
    random_state=42, stratify=y
)

# 4. Normalize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initialize ANN parameters
np.random.seed(42)

input_neurons = 4
hidden_neurons = 10
output_neurons = 3

W1 = np.random.randn(input_neurons, hidden_neurons) * 0.1
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * 0.1
b2 = np.zeros((1, output_neurons))

learning_rate = 0.05
epochs = 3000

# 6. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 7. Backpropagation Training
loss_history = []

for epoch in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X_train, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = softmax(final_input)

    # Calculate Error
    m = X_train.shape[0]
    loss = -np.mean(
        np.sum(y_train * np.log(predicted_output + 1e-9), axis=1)
    )

    loss_history.append(loss)

    # Backward Propagation
    output_error = (predicted_output - y_train) / m

    dW2 = np.dot(hidden_output.T, output_error)
    db2 = np.sum(output_error, axis=0, keepdims=True)

    hidden_error = np.dot(output_error, W2.T)
    hidden_delta = hidden_error * hidden_output * (1 - hidden_output)

    dW1 = np.dot(X_train.T, hidden_delta)
    db1 = np.sum(hidden_delta, axis=0, keepdims=True)

    # Update Weights and Biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 8. Testing the ANN
hidden_test = sigmoid(np.dot(X_test, W1) + b1)
output_test = softmax(np.dot(hidden_test, W2) + b2)

y_pred = np.argmax(output_test, axis=1)
y_actual = np.argmax(y_test, axis=1)

# 9. Evaluate Performance
accuracy = accuracy_score(y_actual, y_pred)

print("\nANN MODEL RESULTS")
print("-------------------------")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_actual, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_actual, y_pred,
    target_names=iris.target_names
))

# 10. Plot Training Loss
plt.plot(loss_history)
plt.title("ANN Backpropagation - Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

print("\nANN Backpropagation Completed Successfully!")
