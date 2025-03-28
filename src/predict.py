from src.train import train

def estimatePrice(mileage: float):
	"""
	Estimate the price depending on the car's mileage.
	:param mileage: the car's mileage.
	:return: none.
	"""
	theta0, theta1 = 0, 0
	train("./data/data.csv")
	with open("./data/metrics.txt", "r") as file:
		lines = file.readlines()
	theta0 = float(lines[0].split("=")[1].strip())
	theta1 = float(lines[1].split("=")[1].strip())
	estimatedPrice = theta0 + (theta1 * mileage)
	return estimatedPrice

def main():
	while True:
		mileage = input("Mileage: ").strip()
		try:
			mileage = float(mileage)
			print("The estimated price is:", estimatePrice(mileage))
			break
		except ValueError:
			print("Please enter a valid mileage.")

if __name__ == "__main__":
	main()