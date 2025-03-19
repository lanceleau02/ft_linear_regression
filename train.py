import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/data.csv')

def model(X, theta0, theta1):
	# Formula: ax + b || theta0 + (theta1 * mileage)
	return theta0 + (theta1 * X)

def cost_function(X, y, theta0, theta1):
	# Formula: (1 / (2 * m)) * sum((f(x) - y) ** 2)
	m = len(y)
	return (1 / (2 * m)) * np.sum((model(X, theta0, theta1) - y) ** 2)

def gradient(X, y, theta0, theta1):
	# Formula: (1 / m) * sum(x * (f(x) - y))
	m = len(y)
	error = model(X, theta0, theta1) - y
	theta0_gradient = (1 / m) * np.sum(error)
	theta1_gradient = (1 / m) * np.sum(error * X) # element-wise multiplication
	return theta0_gradient, theta1_gradient

def gradient_descent(X, y, theta0, theta1, learning_rate, n_iterations):
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations):
		theta0_gradient, theta1_gradient = gradient(X, y, theta0, theta1)
		theta0 = theta0 - learning_rate * theta0_gradient
		theta1 = theta1 - learning_rate * theta1_gradient
		cost_history[i] = cost_function(X, y, theta0, theta1)
	return theta0, theta1, cost_history

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 train.py <datafile>")
		sys.exit(1)

	# Get the values from the dataset
	x = df['km'].to_numpy()
	y = df['price'].to_numpy()

	# Normalize the data
	x_min = x.min()
	x_max = x.max()
	X = (x - x_min) / (x_max - x_min)
	y_min = y.min()
	y_max = y.max()
	Y = (y - y_min) / (y_max - y_min)

	# Gradient descent
	theta0, theta1, cost_history = gradient_descent(X, Y, 0, 0, 0.1, 1000)
	plt.plot(range(1000), cost_history)
	plt.show()

	# Denormalize thetas
	theta1_prime = theta1 * (y.max() - y.min()) / (x.max() - x.min())
	theta0_prime = theta0 * (y.max() - y.min()) + y.min() - theta1_prime * x.min()

	# Display the new model
	predictions = model(x, theta0_prime, theta1_prime)
	plt.scatter(x, y)
	plt.plot(x, predictions, color='red')
	plt.show()

	# Save thetas in a file
	with open("metrics.txt", "w") as file:
		file.write(f"theta0 = {theta0_prime}\n")
		file.write(f"theta1 = {theta1_prime}\n")

	df2 = pd.read_csv('./data/predicted_data.csv')

	df2['predictedPrice'] = df['km'].apply(lambda x: round(theta0_prime + theta1_prime * x))
	df2.to_csv('./data/predicted_data.csv', index=False)

if __name__ == "__main__":
	main()
