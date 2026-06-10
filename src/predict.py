METRICS_FILE = "./data/metrics.txt"
PREDICTED_DATA_FILE = "./data/predicted_data.csv"

def load_thetas():
	try:
		with open(METRICS_FILE, "r") as file:
			lines = file.readlines()
		theta0 = float(lines[0].split("=")[1].strip())
		theta1 = float(lines[1].split("=")[1].strip())
		return theta0, theta1
	except (FileNotFoundError, ValueError, IndexError):
		return 0.0, 0.0

def estimatePrice(mileage: float, theta0: float, theta1: float) -> float:
	"""
	Estimate the price depending on the car's mileage.
	:param mileage: the car's mileage.
	:param theta0: intercept of the line.
	:param theta1: slope of the line.
	:return: estimated price of the car.
	:formula: estimatePrice(mileage) = theta0 + (theta1 * mileage)
	"""
	return theta0 + (theta1 * mileage)

def main():
	theta0, theta1 = load_thetas()
	while True:
		mileage = input("Mileage: ").strip()
		try:
			mileage = float(mileage)
			if mileage <= 0:
				print("Mileage must be a strictly positive number.")
				continue
			price = round(estimatePrice(mileage, theta0, theta1))
			if price < 0:
				print("Warning: mileage is too high, the estimated price is not reliable.")
			else:
				print("The estimated price is:", price)
			break
		except ValueError:
			print("Please enter a valid mileage.")

if __name__ == "__main__":
	main()