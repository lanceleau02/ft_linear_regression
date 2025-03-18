import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

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
	# Create a random values dataset
	x, y = make_regression(n_samples=100, n_features=1, noise=10)

	# Display the dataset
	plt.scatter(x,y)
	plt.show()

	# Print NumPy arrays and reshape y from (100) to (100, 1)
	print(x.shape)
	y = y.reshape(y.shape[0], 1)
	print(y.shape)

	# Add a bias column to x and store it in the design matrix X
	X = np.hstack((x, np.ones(x.shape)))
	print(X.shape)

	# Random initialization of Theta
	theta = np.random.randn(2, 1)
	print(theta)
	
	# Model verification by displaying a linear regression line
	plt.scatter(x,y)
	plt.plot(x, model(X, theta), c='r')
	plt.show()

	# Gradient descent calculation with cost history
	theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000)
	# Display the cost function curve depending on the number of iterations (here 1000)
	plt.plot(range(1000), cost_history)
	plt.show()

	# Display the new model
	predictions = model(X, theta_final)
	plt.scatter(x, y)
	plt.plot(x, predictions, c='r')
	plt.show()

	# Display the rate of the model's accuracy
	print(coef_determination(y, predictions))

if __name__ == "__main__":
	main()