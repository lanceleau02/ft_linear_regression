import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')

def test():
	# Create a matrice X
	X = np.hstack(())

def viewData():
	mileage = df['km'].tolist()
	price = df['price'].tolist()
	plt.scatter(mileage, price, color = 'b')
	plt.show()

def formulas():
	tmpTheta0, tmpTheta1, alpha, m = 0, 0, 0.01, df.shape[0]
	
	for i in range(0, 1000):
		error = tmpTheta0 + (tmpTheta1 * row['km']) - row['price']
		tmpTheta0 -= alpha * (1 / m) * sum(range(0, m)) * error
		tmpTheta1 -= alpha * (1 / m) * sum(range(0, m)) * error * row['km']

def readData(datafile):
	for index, row in df.iterrows():
		print(row['km'], row['price'])
	formulas()

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 train.py <datafile>")
		sys.exit(1)
	readData(sys.argv[1])
	viewData()

if __name__ == "__main__":
	main()