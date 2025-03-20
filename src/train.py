import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/data.csv')

def model(X, theta0, theta1) -> list:
	"""
	Calculates the predicted values (y_pred) from the features (X).
	:param X: features.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: predicted values.
	:formula: f(x) = ax + b = theta0 + (theta1 * mileage)
	"""
	return theta0 + (theta1 * X)

def cost_function(X, y, theta0, theta1) -> list:
	"""
	Measures model's errors by comparing predicted values (y_pred) with true values (y).
	:param X: features.
	:param y: true values.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: mean squared error (MSE).
	:formula: J(a,b) = (1 / (2 * m)) * sum((f(x) - y) ** 2)
	"""
	m = len(y)
	return (1 / (2 * m)) * np.sum((model(X, theta0, theta1) - y) ** 2)

def gradient(X, y, theta0, theta1) -> tuple:
	"""
	Calculates the gradient values.
	:param X: features.
	:param y: true values.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: gradient values.
	:formula: Gradient = (1 / m) * sum(x * (f(x) - y))
	"""
	m = len(y)
	error = model(X, theta0, theta1) - y
	theta0_gradient = (1 / m) * np.sum(error)
	theta1_gradient = (1 / m) * np.sum(error * X)
	return theta0_gradient, theta1_gradient

def gradient_descent(X, y, theta0, theta1, alpha, iterations) -> tuple:
	"""
	Optimizes the model parameters by iteratively minimizing the cost function.
	:param X: features.
	:param y: true values.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:param alpha: learning rate.
	:param iterations: number of iterations.
	:return: thetas and cost history.
	"""
	cost_history = np.zeros(iterations)
	for i in range(0, iterations):
		theta0_gradient, theta1_gradient = gradient(X, y, theta0, theta1)
		theta0 = theta0 - alpha * theta0_gradient
		theta1 = theta1 - alpha * theta1_gradient
		cost_history[i] = cost_function(X, y, theta0, theta1)
	return theta0, theta1, cost_history

def normalize_data(x, y):
	"""
	Normalizes the x and y values using the min-max normalization.
	:param x: features.
	:param y: true values.
	:return: x and y normalized.
	"""
	x_norm = (x - x.min()) / (x.max() - x.min())
	y_norm = (y - y.min()) / (y.max() - y.min())
	return x_norm, y_norm

def train():
	"""
	Trains the model using the gradient descent algorithm.
	:param: none.
	:return: none.
	"""
	# Get the values from the dataset
	x = df['km'].to_numpy()
	y = df['price'].to_numpy()

	# Normalize the data
	x_norm, y_norm = normalize_data(x, y)

	# Gradient descent
	theta0, theta1, cost_history = gradient_descent(x_norm, y_norm, 0, 0, 0.1, 1000)
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

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 train.py <datafile>")
		sys.exit(1)
	train()

if __name__ == "__main__":
	main()
