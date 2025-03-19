from metrics import *
import os.path, sys
import numpy as np

# n      = number of samples
# p      = number of features
# y_true = true value (ground truth)
# y_pred = predicted value
# y_mean = mean of true values

def mean_absolute_error(y_true, y_pred):
	# Formula: (1 / n) * sum(abs(y_true - y_pred))
	return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
	# Formula: (1 / n) * sum((y_true - y_pred) ** 2)
	return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
	# Formula: math.sqrt((1 / n) * sum((y_true - y_pred) ** 2))
	return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred, y_mean):
	# Formula: 1 - (u / v)
	u = np.sum((y_true - y_pred) ** 2)
	v = np.sum((y_true - y_mean) ** 2)
	return 1 - (u / v)

def adjusted_r_squared(n, p, y_true, y_pred, y_mean):
	# Formula: 1 - (1 - R^2) * ((n - 1) / (n - p - 1))
	return 1 - (1 - r_squared(y_true, y_pred, y_mean)) * ((n - 1) / (n - p - 1))

def explained_variance_score(y_true, y_pred):
	# Formula: 1 - (Var(y_true - y_pred) / Var(y_true))
	return 1 - (np.var(y_true - y_pred) / np.var(y_true))

def main():
	if os.path.isfile("metrics.py") == False:
		print("Metrics file not found. Please run the 'train.py' file.")
		sys.exit(1)

	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = root_mean_squared_error(y_true, y_pred)
	r2 = r_squared(y_true, y_pred, np.mean(y_true))
	ar2 = adjusted_r_squared(n, p, y_true, y_pred, np.mean(y_true))
	evs = explained_variance_score(y_true, y_pred)

	print(f"Mean Absolute Error:            {mae}")
	print(f"Mean Squared Error:             {mse}")
	print(f"Root Mean Squared Error:        {rmse}")
	print(f"R-squared:                      {r2}")
	print(f"Adjusted R-squared:             {ar2}")
	print(f"Explained Variance Score:       {evs}")

if __name__ == "__main__":
	main()
