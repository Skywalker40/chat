import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([2, 3, 5, 7, 9])
y = np.array([4, 5, 7, 10, 15])

# Initialize parameters
theta0 = 0
theta1 = 0
alpha = 0.01
epochs = 1000
m = len(X)

# Gradient Descent
for _ in range(epochs):
    h = theta0 + theta1 * X
    error = h - y
    theta0 -= alpha * (1/m) * np.sum(error)
    theta1 -= alpha * (1/m) * np.sum(error * X)

# Final model
print(f"Model: y = {theta1:.2f}x + {theta0:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, theta0 + theta1 * X, color='red', label='Predicted')
plt.xlabel('Sunlight (hours)')
plt.ylabel('Plant Height (cm)')
plt.legend()
plt.show()