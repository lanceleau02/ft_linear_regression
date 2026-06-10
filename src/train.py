import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_FILE = "./data/metrics.txt"
PREDICTED_DATA_FILE = "./data/predicted_data.csv"

def model(X, theta0, theta1) -> np.ndarray:
	"""
	Calculates the predicted values (y_pred) from the features (X).
	:param X: features.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: predicted values.
	:formula: f(x) = ax + b = theta0 + (theta1 * mileage)
	"""
	return theta0 + (theta1 * X)

def cost_function(X, y, theta0, theta1) -> float:
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

def get_data(datafile):
	"""
	Get x (features) and y (true values) values from the dataset.
	:param datafile: the datafile.
	:return: x and y.
	"""
	df = pd.read_csv(datafile)
	x = df['km'].to_numpy()
	y = df['price'].to_numpy()
	return x, y, df

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

def denormalize_thetas(x, y, theta0, theta1):
	"""
	Denormalizes the two thetas.
	:param x: features.
	:param y: true values.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: the two denormalized thetas.
	"""
	theta1_denorm = theta1 * (y.max() - y.min()) / (x.max() - x.min())
	theta0_denorm = theta0 * (y.max() - y.min()) + y.min() - theta1_denorm * x.min()
	return theta0_denorm, theta1_denorm

def save_thetas(theta0, theta1):
	"""
	Save the two thetas in a .txt file.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: none.
	"""
	with open(METRICS_FILE, "w") as file:
		file.write(f"theta0 = {theta0}\n")
		file.write(f"theta1 = {theta1}\n")

def save_predicted_price(theta0, theta1, df):
	""" 
	Save the predicted price for each mileage in a .csv file.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: none.
	"""
	df_out = df[['km', 'price']].copy()
	df_out['predictedPrice'] = df['km'].apply(lambda x: round(theta0 + theta1 * x))
	df_out.to_csv(PREDICTED_DATA_FILE, index=False)

def train(datafile, alpha, iterations):
	"""
	Trains the model using the gradient descent algorithm.
	:param datafile: the datafile.
	:return: none.
	"""
	x, y, df = get_data(datafile)

	x_norm, y_norm = normalize_data(x, y)

	theta0, theta1, cost_history = gradient_descent(x_norm, y_norm, 0, 0, alpha, iterations)

	theta0_denorm, theta1_denorm = denormalize_thetas(x, y, theta0, theta1)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	ax1.plot(range(iterations), cost_history)
	ax1.set_title("Cost history")
	ax1.set_xlabel("Iterations")
	ax1.set_ylabel("Cost")
	ax2.set_xlabel("Mileage (km)")
	ax2.set_ylabel("Price")
	ax2.scatter(x, y, label="Data")
	ax2.plot(x, model(x, theta0_denorm, theta1_denorm), color='red', label="Regression")
	ax2.legend()
	plt.tight_layout()
	plt.show()

	save_thetas(theta0_denorm, theta1_denorm)
	save_predicted_price(theta0_denorm, theta1_denorm, df)

def main():
	if len(sys.argv) != 4:
		print('Usage: train.py <datafile> <learning_rate> <iterations>')
		sys.exit()
	try:
		datafile = sys.argv[1]
		alpha = float(sys.argv[2])
		iterations = int(sys.argv[3])
		if alpha <= 0 or iterations <= 0:
			print('Invalid arguments. Learning rate and iterations must be positive.')
			sys.exit()
		if not os.path.exists(datafile):
			print('Invalid arguments. Datafile does not exist.')
			sys.exit()
	except ValueError:
		print('Invalid arguments. Please provide a valid learning rate and number of iterations.')
		sys.exit()
	train(datafile, alpha, iterations)

if __name__ == "__main__":
	main()
