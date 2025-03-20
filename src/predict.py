from src.train import train

def estimatePrice(mileage):
	theta0, theta1 = 0, 0
	train()
	with open("metrics.txt", "r") as file:
		lines = file.readlines()
	theta0 = float(lines[0].split("=")[1].strip())
	theta1 = float(lines[1].split("=")[1].strip())
	estimatedPrice = theta0 + (theta1 * mileage)
	print(estimatedPrice)

def main():
	mileage = int(input("Mileage: "))
	estimatePrice(mileage)

if __name__ == "__main__":
	main()