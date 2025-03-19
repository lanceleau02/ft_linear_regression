def estimatePrice(mileage):
	theta0, theta1 = 0, 0
	estimatedPrice = theta0 + (theta1 * mileage)
	print(estimatedPrice)

def main():
	mileage = int(input("Mileage: "))
	estimatePrice(mileage)

if __name__ == "__main__":
	main()