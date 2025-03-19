import os.path, sys
import numpy as np

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
	if os.path.isfile("metrics.txt") == False:
		print("Metrics file not found. Please run the 'train.py' file.")
		sys.exit(1)

	with open("metrics.txt", "r") as file:
		lines = file.readlines()
	theta0 = float(lines[0].split("=")[1].strip())
	theta1 = float(lines[1].split("=")[1].strip())

	mae = mean_absolute_error(theta0, theta1)
	mse = mean_squared_error(theta0, theta1)
	rmse = root_mean_squared_error(theta0, theta1)
	r2 = r_squared(theta0, theta1)
	ar2 = adjusted_r_squared(theta0, theta1)
	evs = explained_variance_score(theta0, theta1)

	print(f"Mean Absolute Error:            {mae}")
	print(f"Mean Squared Error:             {mse}")
	print(f"Root Mean Squared Error:        {rmse}")
	print(f"R-squared:                      {r2}")
	print(f"Adjusted R-squared:             {ar2}")
	print(f"Explained Variance Score:       {evs}")

if __name__ == "__main__":
	main()
