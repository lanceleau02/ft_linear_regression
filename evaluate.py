import os.path
import numpy as np

# n      = number of samples
# p      = number of features
# y_true = true value (ground truth)
# y_pred = predicted value
# y_mean = mean of true values

def mean_absolute_error(y_true, y_pred):
	# Formula: (1 / n) * sum(abs(y_true - y_pred))
	return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
	# Formula: (1 / n) * sum(abs((y_true - y_pred) / y_true) * 100)
	return np.mean(abs((y_true - y_pred) / y_true) * 100)

def mean_squared_error(y_true, y_pred):
	# Formula: (1 / n) * sum((y_true - y_pred) ** 2)
	return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
	# Formula: math.sqrt((1 / n) * sum((y_true - y_pred) ** 2))
	return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred, y_mean):
	# Formula: 1 - (u / v)
	u = sum((y_true - y_pred) ** 2)
	v = sum((y_true - y_mean) ** 2)
	return 1 - (u / v)

def adjusted_r_squared(n, p, y_true, y_pred, y_mean):
	# Formula: 1 - (1 - R^2) * ((n - 1) / (n - p - 1))
	return 1 - (1 - r_squared(y_true, y_pred, y_mean)) * ((n - 1) / (n - p - 1))

def explained_variance_score(y_true, y_pred):
	# Formula: 1 - (Var(y_true - y_pred) / Var(y_true))
	return 1 - (np.var(y_true - y_pred) / np.var(y_true))

def main():
	if os.path.isfile("metrics.txt") == False:
		print(" ")

if __name__ == "__main__":
	main()
