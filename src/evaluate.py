import os.path, sys
import numpy as np
import pandas as pd

def mean_absolute_error(y_true, y_pred):
	"""
	Calculates the Mean Absolute Error (MAE).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Mean Absolute Error (MAE).
	:formula: MAE = (1 / n) * sum(abs(y_true - y_pred))
	"""
	return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
	"""
	Calculates the Mean Squared Error (MSE).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Mean Squared Error (MAE).
	:formula: MSE = (1 / n) * sum((y_true - y_pred) ** 2)
	"""
	return np.mean((y_true - y_pred) ** 2)

def mean_squared_percentage_error(y_true, y_pred):
	"""
	Calculates the Mean Squared Percentage Error (MSPE).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Mean Squared Percentage Error (MSPE).
	:formula: MAPE = (1 / n) * sum(abs((y_true - y_pred) / y_true) * 100)
	"""
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
	"""
	Calculates the Root Mean Squared Error (RMSE).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Root Mean Squared Error (RMSE).
	:formula: RMSE = sqrt((1 / n) * sum((y_true - y_pred) ** 2))
	"""
	return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
	"""
	Calculates the R-squared (R^2).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the R-squared (R^2).
	:formula: R^2 = 1 - (u / v)
	"""
	u = np.sum((y_true - y_pred) ** 2)
	v = np.sum((y_true - np.mean(y_true)) ** 2)
	return 1 - (u / v)

def adjusted_r_squared(y_true, y_pred):
	"""
	Calculates the Adjusted R-squared (R^2).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Adjusted R-squared (R^2).
	:formula: Adjusted R^2 = 1 - (1 - R^2) * ((n - 1) / (n - p - 1))
	"""
	return 1 - (1 - r_squared(y_true, y_pred)) * ((len(y_true) - 1) / (len(y_true) - 1 - 1))

def explained_variance_score(y_true, y_pred):
	"""
	Calculates the Explained Variance Score (EVS).
	:param y_true: true values.
	:param y_pred: predicted values.
	:return: the Explained Variance Score (EVS).
	:formula: EVS = 1 - (Var(y_true - y_pred) / Var(y_true))
	"""
	return 1 - (np.var(y_true - y_pred) / np.var(y_true))

def main():
	if os.path.isfile("metrics.txt") == False:
		print("Metrics file not found. Please run the 'train.py' file.")
		sys.exit(1)

	df = pd.read_csv('./data/pred_data.csv')
	y_true = df['price'].to_numpy()
	y_pred = df['predictedPrice'].to_numpy()

	print("Mean Absolute Error:            {:.2f}".format(mean_absolute_error(y_true, y_pred)))
	print("Mean Squared Error:             {:.2f}".format(mean_squared_error(y_true, y_pred)))
	print("Mean Squared Percentage Error:  {:.2f}%".format(mean_squared_percentage_error(y_true, y_pred)))
	print("Root Mean Squared Error:        {:.2f}".format(root_mean_squared_error(y_true, y_pred)))
	print("R-squared:                      {:.2f}".format(r_squared(y_true, y_pred)))
	print("Adjusted R-squared:             {:.2f}".format(adjusted_r_squared(y_true, y_pred)))
	print("Explained Variance Score:       {:.2f}".format(explained_variance_score(y_true, y_pred)))

if __name__ == "__main__":
	main()
