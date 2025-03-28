<div align="center">

# ft_linear_regression

**An introduction to machine learning.**

</div>

## Features

- Implement a simple linear regression model from scratch.
- Load and preprocess a dataset for training and evaluation.
- Train the model using gradient descent to minimize error.
- Save and load model parameters for future predictions.
- Predict car prices based on mileage using the trained model.
- Visualize data with Matplotlib to analyze results.
- Evaluate model performance using error metrics.
- Handle user input for mileage and return estimated prices.
- Ensure numerical stability and avoid common pitfalls in regression.
- Optimize hyperparameters like learning rate for better convergence.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lanceleau02/ft_linear_regression.git
```

2. Navigate to the project directory:

```bash
cd ft_linear_regression
```

3. Create and install the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

You first need to run the `train` program:

```bash
python3 -m src.train
```

Then, you can run the `predict` and `evaluate` programs:

```bash
python3 -m src.predict
```

```bash
python3 -m src.evaluate
```

Note: the `predict` program can be run without running the `train` program first.

## License

This project is licensed under the **42 School** License.

- **Educational Use Only**: This project is intended for educational purposes at the 42 School as part of the curriculum.
- **Non-commercial Use**: The code may not be used for commercial purposes or redistributed outside of the 42 School context.
- **No Warranty**: The project is provided "as-is", without any warranty of any kind.

For more details, see the [LICENSE](https://github.com/lanceleau02/ft_linear_regression/blob/main/LICENSE) file.

## Resources

- [Machine Learning course in French (YouTube)](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)

