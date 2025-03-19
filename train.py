import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')

def viewData():
	mileage = df['km'].tolist()
	price = df['price'].tolist()
	# plt.scatter(mileage, price, color = 'b')
	# plt.show()

def model(X, theta):
	return X.dot(theta)

def cost_function(X, y, theta):
	m = len(y)
	return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)

def grad(X, y, theta):
	m = len(y)
	return 1 / m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations):
		theta = theta - learning_rate * grad(X, y, theta)
		cost_history[i] = cost_function(X, y, theta)
	return theta, cost_history

def coef_determination(y, pred):
	u = ((y - pred) ** 2).sum()
	v = ((y - y.mean()) ** 2).sum()
	return 1 - u / v

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 train.py <datafile>")
		sys.exit(1)
	# readData(sys.argv[1])

	# Get the values from the dataset
	x = df['km'].to_numpy()
	y = df['price'].to_numpy()
	# Standardize the data
	scaler = StandardScaler()
	x = scaler.fit_transform(x.reshape(-1, 1))
	y = scaler.fit_transform(y.reshape(-1, 1))
	# Reshape the arrays
	x = x.reshape(x.shape[0], 1)
	y = y.reshape(y.shape[0], 1)

	# Display the dataset
	plt.scatter(x,y)
	plt.show()

	# Add a bias to x array and store it in X matrix
	X = np.hstack((x, np.ones(x.shape)))

	theta, cost_history = gradient_descent(X, y, np.zeros((2,1)), 0.01, 1000)
	plt.plot(range(1000), cost_history)
	plt.show()

	# Display the new model
	predictions = model(X, theta)
	plt.scatter(x, y)
	plt.plot(x, predictions, c='r')
	plt.show()

	print(coef_determination(y, predictions))

	f = open("metrics.txt", "w")
	f.write(f"n = {len(x)}\n")
	f.write(f"p = {1}\n")
	f.write(f"y_true = {np.array(y)}\n")
	f.write(f"y_pred = {np.array(predictions)}\n")
	f.write(f"y_mean = {np.mean(y)}\n")
	f.close()

if __name__ == "__main__":
	main()