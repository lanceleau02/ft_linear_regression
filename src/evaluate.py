import os.path, sys
import numpy as np
import pandas as pd

def mean_absolute_error(y_true, y_pred):
	# Formula: (1 / n) * sum(abs(y_true - y_pred))
	return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
	# Formula: (1 / n) * sum((y_true - y_pred) ** 2)
	return np.mean((y_true - y_pred) ** 2)

def mean_squared_percentage_error(y_true, y_pred):
	# Formula: (1 / n) * sum(abs((y_true - y_pred) / y_true) * 100)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
	# Formula: math.sqrt((1 / n) * sum((y_true - y_pred) ** 2))
	return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
	# Formula: 1 - (u / v)
	u = np.sum((y_true - y_pred) ** 2)
	v = np.sum((y_true - np.mean(y_true)) ** 2)
	return 1 - (u / v)

def adjusted_r_squared(y_true, y_pred):
	# Formula: 1 - (1 - R^2) * ((n - 1) / (n - p - 1))
	return 1 - (1 - r_squared(y_true, y_pred)) * ((len(y_true) - 1) / (len(y_true) - 1 - 1))

def explained_variance_score(y_true, y_pred):
	# Formula: 1 - (Var(y_true - y_pred) / Var(y_true))
	return 1 - (np.var(y_true - y_pred) / np.var(y_true))

def main():
	if os.path.isfile("metrics.txt") == False:
		print("Metrics file not found. Please run the 'train.py' file.")
		sys.exit(1)

	df = pd.read_csv('./data/predicted_data.csv')

	# Get the values from the dataset
	y_true = df['price'].to_numpy()
	y_pred = df['predictedPrice'].to_numpy()

	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	mspe = mean_squared_percentage_error(y_true, y_pred)
	rmse = root_mean_squared_error(y_true, y_pred)
	r2 = r_squared(y_true, y_pred)
	ar2 = adjusted_r_squared(y_true, y_pred)
	evs = explained_variance_score(y_true, y_pred)

	print("Mean Absolute Error:            {:.2f}".format(mae))
	print("Mean Squared Error:             {:.2f}".format(mse))
	print("Mean Squared Percentage Error:  {:.2f}%".format(mspe))
	print("Root Mean Squared Error:        {:.2f}".format(rmse))
	print("R-squared:                      {:.2f}".format(r2))
	print("Adjusted R-squared:             {:.2f}".format(ar2))
	print("Explained Variance Score:       {:.2f}".format(evs))

if __name__ == "__main__":
	main()
