import sys
import pandas as pd

df = pd.read_csv('data.csv')

def formulas():
	alpha, m = 0.90, df.shape[0]
	
	for index, row in df.iterrows():
		tmpTheta0 = alpha * (1 / m) * sum(0, (m - 1)) * (tmpTheta0 + (tmpTheta1 * row['km']), row['price'])
		tmpTheta1 = alpha * (1 / m) * sum(0, (m - 1)) * (tmpTheta0 + (tmpTheta1 * row['km']), row['price']) * row['km']

def readData(datafile):
	for index, row in df.iterrows():
		print(row['km'], row['price'])
	formulas()

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 train.py <datafile>")
		sys.exit(1)
	readData(sys.argv[1])

if __name__ == "__main__":
	main()