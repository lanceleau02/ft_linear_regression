<div align="center">

# ft_linear_regression

**An introduction to machine learning.**

</div>

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
python3 -m src.train <datafile> <learning_rate> <iterations>
```

Then, you can run the `predict` and `evaluate` programs:

```bash
python3 -m src.predict
```

```bash
python3 -m src.evaluate
```

Note: the `predict` program can be run without running the `train` program first.

## Subject Breakdown

This project introduces machine learning by implementing a simple **linear regression** model from scratch. The goal is to predict **car prices based on mileage** using a dataset of past sales. It consists in 3 files: `train.py` to train the model, `predict.py` to predict a price depending on a given mileage and based on the data from the dataset and, finally, the `evaluate.py` file to evaluate the model's performance thanks to different formulas (MAE, MSE, R-squared...).

At its core, the model finds patterns in data: cars with more mileage tend to have lower prices. Instead of manually analyzing these trends, the model learns the relationship between mileage and price, then makes predictions for new cars.

The process starts with training the model on a dataset of car prices and mileage. Using **gradient descent**, it adjusts its parameters to minimize prediction errors. Once trained, the model can estimate the price of a car based on its mileage, and its accuracy is evaluated using error metrics. This project covers the essential steps of supervised learning and introduces fundamental concepts used in more complex machine learning models.

## Approach & Implementation

To breakdown the approach, we'll follow the 4 following steps:

1. **Import a dataset** containing **features** ($x$) and a **target** ($y$).
2. **Develop a model** with **parameters** that the machine must learn.
3. **Define a cost function**, which measures the model's error.
4. **Optimize this cost function** using a learning algorithm.

### Dataset

- A **dataset** always contains two types of variable:
	- **Target variable (y)**: what the machine must predict (e.g. car price).
    - **Features ($x_1$, $x_2$, $x_3$, ...)**: factors influencing **y** (e.g. mileage, age, brand).
- Notations :
	- **m** = number of **examples** (dataset rows).
	- **n** = number of **features** (dataset columns, excluding y).
- Datasets are often represented as **matrices and vectors** in machine learning.

The dataset used in this implementation consists of data points with two main attributes for each car: **mileage** (`km`) and **price** (`price`). This dataset is stored in a CSV file, where each row represents one car. The mileage is the feature (independent variable), and the price is the target value (dependent variable).

In the code, the `get_data` function retrieves this data from the CSV file. It extracts the **mileage** values into the `x` array and the **price** values into the `y` array, which are used for training the model.

Mathematically, the dataset consists of pairs of input-output values $(x_i, y_i)$ for each car, where:

- $x_i$ is the mileage of the $i$-th car,
- $y_i$ is the price of the $i$-th car,

The task is to use the mileage values $x$ to predict the price $y$ using a linear model.

### Model

- A **model** is a function that relates the features to the target variable.
- Each model has **parameters** (coefficients) that the machine must learn.
- It's the human who chooses which type of model to use, and the machine learns the optimal parameters.
- A good model is a generalization of the data.

The **model** in this case is a **linear regression model**, defined by the equation:

$$
f(x) = \theta_0 + \theta_1 \cdot X
$$

Where:
- $f(x)$ is the predicted price,
- $X$ is the feature (mileage),
- $\theta_0$ is the **intercept** (the predicted price when the mileage is 0),
- $\theta_1$ is the **slope** (how much the price changes per unit of mileage).

The `model` function computes these predictions for a given set of feature values $X$, using the current values of $\theta_0$ and $\theta_1$:

```python
def model(X, theta0, theta1):
    return theta0 + (theta1 * X)
```

At the start of training, $\theta_0$ and $\theta_1$ are typically initialized to 0 (or small random values), and the learning algorithm will adjust them over time to improve predictions.

### Cost Function

- A **cost function** measures model errors by comparing predictions with true dataset values.
- Objective: **Minimize** these errors to obtain an accurate model.
- Example:
    - If a car with 100,000 km is actually valued at **$5,000** but the model predicts **$7,000**, the error is **$2,000**.
    - These errors are added together to obtain the **global cost function**.

In this implementation, the **Mean Squared Error (MSE)** is used as the cost function, which is commonly used for regression tasks. The formula for MSE is:

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (f(x_i) - y_i)^2
$$

Where:
- $m$ is the number of data points (cars),
- $f(x_i)$ is the predicted price for the $i$-th car based on the current model parameters,
- $y_i$ is the true price for the $i$-th car.

The `cost_function` function in the code implements this formula. It calculates the squared differences between the predicted prices and the actual prices for all data points, averages them, and divides by 2 to simplify the calculation of gradients later:

```python
def cost_function(X, y, theta0, theta1):
    m = len(y)
    return (1 / (2 * m)) * np.sum((model(X, theta0, theta1) - y) ** 2)
```

The goal of the learning algorithm is to minimize this cost function, meaning we want to find the values of $\theta_0$ and $\theta_1$ that result in the smallest possible error between the predicted and actual prices.

### Learning Algorithm (Gradient Descent)

A **learning algorithm** adjusts the model parameters to minimize the cost function. 

Imagine being lost in the mountains and wanting to reach the lowest point without a map. At each step, you identify where the slope is steepest and move in that direction. Repeat until you reach the bottom of the valley. This is exactly how **Gradient Descent** works.

It computes the gradients (partial derivatives) of the cost function with respect to each parameter, and then updates the parameters in the direction that reduces the cost:

$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)
$$

$$
\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i) \cdot x_i
$$

The `gradient` function computes these gradients:

```python
def gradient(X, y, theta0, theta1):
    m = len(y)
    error = model(X, theta0, theta1) - y
    theta0_gradient = (1 / m) * np.sum(error)
    theta1_gradient = (1 / m) * np.sum(error * X)
    return theta0_gradient, theta1_gradient
```

After calculating the gradients, we update the parameters by subtracting a fraction of the gradient. The fraction is determined by the **learning rate** ($\alpha$). If $\alpha$ is too large, the model risks oscillating without reaching the minimum. If it is too small, convergence will be extremely slow.

$$
\theta_0 = \theta_0 - \alpha \cdot \frac{\partial J}{\partial \theta_0}
$$
$$
\theta_1 = \theta_1 - \alpha \cdot \frac{\partial J}{\partial \theta_1}
$$

The `gradient_descent` function implements these updates iteratively:

```python
def gradient_descent(X, y, theta0, theta1, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        theta0_gradient, theta1_gradient = gradient(X, y, theta0, theta1)
        theta0 = theta0 - alpha * theta0_gradient
        theta1 = theta1 - alpha * theta1_gradient
        cost_history[i] = cost_function(X, y, theta0, theta1)
    return theta0, theta1, cost_history
```

### Additional Data Handling: Normalization and Denormalization

Before training the model, the data is **normalized** using **min-max normalization** in the `normalize_data` function. This scales the data between 0 and 1, which helps the gradient descent algorithm converge more quickly:

```python
def normalize_data(x, y):
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    return x_norm, y_norm
```

After the model is trained, the parameters $\theta_0$ and $\theta_1$ are **denormalized** back to their original scale using the `denormalize_thetas` function:

```python
def denormalize_thetas(x, y, theta0, theta1):
    theta1_denorm = theta1 * (y.max() - y.min()) / (x.max() - x.min())
    theta0_denorm = theta0 * (y.max() - y.min()) + y.min() - theta1_denorm * x.min()
    return theta0_denorm, theta1_denorm
```

This step ensures that the trained model is capable of making real-world predictions on the original scale of the data.

### Coordinating the Training Process
The `train` function coordinates the entire process: it loads the data, normalizes it, initializes the parameters, runs the gradient descent optimization, and plots the cost history and final regression line. The trained parameters are then saved using `save_thetas`, and the predicted prices for each mileage entry are saved in a CSV file using `save_predicted_price`.

```python
def train(datafile):
    x, y = get_data(datafile)
    x_norm, y_norm = normalize_data(x, y)
    theta0, theta1, cost_history = gradient_descent(x_norm, y_norm, 0, 0, 0.08, 1000)
    plt.plot(range(1000), cost_history)
    plt.show()
    theta0_denorm, theta1_denorm = denormalize_thetas(x, y, theta0, theta1)
    predictions = model(x, theta0_denorm, theta1_denorm)
    plt.scatter(x, y)
    plt.plot(x, predictions, color='red')
    plt.show()
    save_thetas(theta0_denorm, theta1_denorm)
    save_predicted_price(theta0_denorm, theta1_denorm)
```

## License

This project is licensed under the **42 School** License.

- **Educational Use Only**: This project is intended for educational purposes at the 42 School as part of the curriculum.
- **Non-commercial Use**: The code may not be used for commercial purposes or redistributed outside of the 42 School context.
- **No Warranty**: The project is provided "as-is", without any warranty of any kind.

For more details, see the [LICENSE](https://github.com/lanceleau02/ft_linear_regression/blob/main/LICENSE) file.

## Resources

- [Machine Learning course in French (YouTube)](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)

