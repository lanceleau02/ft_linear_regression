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

## Subject Breakdown

This project introduces machine learning by implementing a simple **linear regression** model from scratch. The goal is to predict **car prices based on mileage** using a dataset of past sales. It consists in 3 files: `train.py` to train the model, `predict.py` to predict a price depending on a given mileage and based on the data from the dataset and, finally, the `evaluate.py` file to evaluate the model's performance thanks to different formulas (MAE, MSE, R-squared...).

At its core, the model finds patterns in data: cars with more mileage tend to have lower prices. Instead of manually analyzing these trends, the model learns the relationship between mileage and price, then makes predictions for new cars.

The process starts with training the model on a dataset of car prices and mileage. Using **gradient descent**, it adjusts its parameters to minimize prediction errors. Once trained, the model can estimate the price of a car based on its mileage, and its accuracy is evaluated using error metrics. This project covers the essential steps of supervised learning and introduces fundamental concepts used in more complex machine learning models.

But first, we need to understand what linear regression is.

## Introduction to Linear Regression

### I - What is Machine Learning?

**Machine learning (ML)** is a technology that allows machines to **learn** and **make decisions** without explicit programming, impacting daily tools like Google Search, Facebook facial recognition, and recommendations on platforms like YouTube and Netflix. ML is **revolutionary** because it can handle **vast amounts of data** across sectors like transportation (self-driving cars), healthcare (cancer diagnosis), finance, and more. It typically works by **learning from examples**, with supervised learning being the most common approach. ML problems can be divided into **regression** (predicting continuous variables) and **classification** (predicting discrete variables). Additionally, unsupervised learning allows systems to **learn without predefined examples**, much like learning a language through immersion.

### II - Supervised Learning: the 4 steps.

This project is all about supervised learning. Here are the 4 steps to a successful supervised learning project.

#### 1. The concept of Dataset

- A **dataset** always contains two types of variable:
		- **Target variable (y)**: what the machine must predict (e.g. apartment price, spam detection).
		- **Features ($x_1$, $x_2$, $x_3$, ...)**: factors influencing **y** (e.g. apartment surface, quality, address).
- Notations :
		- **m** = number of **examples** (dataset rows).
		- **n** = number of **features** (dataset columns, excluding y).
- Datasets are often represented as **matrices and vectors** in machine learning.

**Example for a dataset on apartments:**

<div align="center">

| Price        | Surface      | Quality      | Zip Code     |
| :---         | :---         | :---         | :---         |
| 313,000      | 90           | 3            | 95000        |
| 720,000      | 110          | 5            | 93000        |
| 250,000      | 40           | 4            | 44500        |
| 290,000      | 60           | 3            | 67000        |
| 190,000      | 50           | 3            | 59300        |
| ...          | ...          | ...          | ...          |

</div>

Where the **Price** column represents the **Target (y)** and the **Surface**, **Quality** and **Zip Code** columns represent the **Features (x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>)**.

By convention, we note **m (number of examples)** and **n (number of features)**.

#### 2. The concept of Model

- A **model** is a function that relates the features to the target variable, e.g. :
		- **Linear model**: $y=ax+b$
		- **Polynomial model**: $y=ax^2+bx+c$
- Each model has **parameters** (coefficients a, b, c...) that the machine must learn.
- It's the human who chooses which type of model to use, and the machine learns the optimal parameters.
- A good model is a generalization.

#### 3. The notion of Cost Function

- A **cost function** measures model errors by comparing predictions with true dataset values.
- Objective: **Minimize** these errors to obtain an accurate model.
- Example:
		- If a 150 m² apartment is valued at **$200,000** but the model predicts **$300,000**, the error is **$100,000**.
		- These errors are added together to obtain the **global cost function**.

#### 4. The notion of Learning Algorithm

- A **learning algorithm** adjusts the model parameters to minimize the cost function.
- Example: **Gradient Descent**, one of the most widely used algorithms.
- The machine adjusts parameters to reduce errors and improve prediction accuracy.

### III - Linear Regression

Linear regression is often underestimated, but it is based on the same concepts as more advanced models (speech recognition, computer vision, etc.).

#### **1. The 4 Key Steps in Supervised Machine Learning**

1. **Import a dataset** containing **features** ($x$) and a **target** ($y$).
2. **Develop a model** with **parameters** that the machine must learn.
3. **Define a cost function**, which measures the model's error.
4. **Optimize this cost function** using a learning algorithm.

#### **2. Application to Linear Regression**

- Example: a dataset of 6 samples with a single feature $x$.
- The chosen model is: $f(x) = ax + b$ (affine function).
- Initially, the parameters $a$ and $b$ are random, and then the machine gradually adjusts them.

#### **3. Definition of the Cost Function**

- The error between the model's prediction and the true value of $y$ is measured.
- We use the **sum of squared errors** (MSE - Mean Squared Error) to avoid negative values:

$$
J(a,b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x_i) - y_i)^2
$$

- The goal is to **minimize this cost function**.

#### **4. Optimization Methods**

1. **Least Squares Method** (Normal Equations):
		- Finds the optimal solution directly by setting the derivative to zero.
		- Seeks the point $a$ where the tangent to $J$ is horizontal (zero slope), where the derivative becomes zero: $\frac{\partial J}{\partial a} = 0$
		- Fast for small datasets but computationally expensive for large ones.
2. **Gradient Descent**:
		- Starts with a random initial point.
		- Calculates the slope (derivative of the cost function).
		- Takes small steps in the direction of the descending slope until a minimum is reached.
		- Better suited for **large datasets**.

### IV - Gradient Descent

Gradient descent is a key optimization algorithm in **machine learning** and **deep learning**, used to find the **minimum of a convex function**. It plays a crucial role in **supervised learning** by minimizing the **cost function** (e.g., mean squared error), leading to the **best learning model** for tasks like facial recognition, speech recognition, and stock market predictions.

#### **1. Lost in the Mountains**  

- Imagine being lost in the mountains and wanting to reach the **lowest point** without a map.  
- Strategy: at each step, identify **where the slope is steepest** and move in that direction.  
- Repeat until reaching the bottom of the valley.  
- This is exactly how **gradient descent** works.  

#### **2. The Gradient Descent Algorithm**  

- **Model parameters** (e.g., $a$ and $b$) are adjusted to minimize the cost function.  
- Compute the **slope (partial derivative)** to determine the direction for parameter adjustment:  

$$
\frac{\partial J(a_0)}{\partial a}
$$  

- Apply a **learning rate ($α$)** to make a small adjustment.  
- Repeat the process (iteration) until reaching the **global minimum**.  
- The algorithm also works in **3D**, like a topographic map.  

#### **3. Parameter Update Formula**  

- **Update formula** for each parameter:  

$$
a_{i+1}=a_i−α\frac{\partial J(a_i)}{\partial a}
$$  

- The same applies to $b$.  
- This **iterative process** gradually adjusts parameter values.  
- When the curve's slope is **descending**, the derivative ($\frac{\partial J(a_i)}{\partial a}$) is **negative**; when ascending, the derivative is **positive**:  
		- **Descending slope:** $a_i > a_{i + 1}$  
		- **Ascending slope:** $a_i < a_{i + 1}$  
- The **distance** between $a_i$ and $a_{i+1}$ equals the **product** of the **learning rate** and the **derivative**.  
- Important note: the hyperparameter Alpha is always **positive**!  

#### **4. Influence of Learning Rate ($α$)**  

- **If $α$ is too large** → risk of oscillations without reaching the minimum.  
- **If $α$ is too small** → extremely slow convergence.  
- No **magic formula**, you need to **experiment with different values**.  

#### **5. Gradient Calculation** 

- In the formulas $a = a - \alpha \frac{\partial J}{\partial a}$ and $b = b - \alpha \frac{\partial J}{\partial b}$, we must calculate the gradient by computing the partial derivatives ($\frac{\partial J}{\partial a}$ and $\frac{\partial J}{\partial b}$).  
- For mean squared error (MSE) as the cost function:  

$$
J(a,b) = \frac{1}{2m} \sum (ax + b - y)^2
$$  

- The partial derivatives are:  

$$
\frac{\partial J}{\partial a} = \frac{1}{m} \sum x(ax+b-y)
$$  

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum (ax+b-y)
$$  

- For MSE, the general formulas are:  

$$
\frac{\partial J(a,b)}{\partial a} = \frac{1}{m} \sum_{i=1}^m (ax^{(i)} + b - y^{(i)}) x^{(i)}
$$  

$$
\frac{\partial J(a,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m (ax^{(i)} + b - y^{(i)})
$$  

- These formulas **update the parameters** and train the model.

#### 6. Formulas Summary

- **Dataset:** $(x,y)$ with $m$ examples.
- **Model:** $f(x) = ax + b$
- **Cost Function:** $J(a,b) = \frac{1}{2m} \sum_{i=1}^m (ax^{(i)} + b - y^{(i)})^2$
- **Gradients:** $\frac{\partial J(a,b)}{\partial a} = \frac{1}{m} \sum_{i=1}^m x^{(i)} (ax^{(i)} + b - y^{(i)})$ and $\frac{\partial J(a,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m (ax^{(i)} + b - y^{(i)})$
- **Gradient Descent Algorithm:** $a = a - \alpha \frac{\partial J(a,b)}{\partial a}$ and $b = b - \alpha \frac{\partial J(a,b)}{\partial b}$

### IV - The importance of matrix

Matrix notation simplifies calculations, enables more complex models like polynomial regression, and is essential for programming in **Python, TensorFlow, etc.**

#### **1. Representing the Model in Matrix Form**

- **Input data:** $X$, $Y$, where $X$ is a matrix of size $m \times (n+1)$ with a column of ones (bias).  
- **Linear model:**  

$$ f(X) = X\Theta $$  

where $\Theta$ is a vector containing the parameters (a and b for a linear model).  

#### **2. Cost Function in Matrix Form**  
- The cost function is a **scalar**, expressed as:  

$$ J(\Theta) = \frac{1}{2m} \sum (X\Theta - Y)^2 $$  

- All errors are grouped into a vector, and the squared sum is computed.  

#### **3. Gradient Descent in Matrix Form**  
- The parameter update algorithm is given by:  

$$ \frac{\partial J (\Theta)}{\partial \Theta} = \frac{1}{m}X^T(X\Theta - Y) $$  

- $X^T$ ensures matrix dimensions align, generalizing gradient calculations.  

- The **parameter update rule** in gradient descent is:  

$$ \Theta = \Theta - \alpha \frac{\partial J}{\partial \Theta} $$  

where:

- **$\Theta$** is the parameter vector (e.g., $a$ and $b$).  
- **$\alpha$** is the **learning rate**.  
- **$\nabla_{\Theta} J$** is the computed gradient vector.  

#### **4. Extension to Polynomial Regression**  
- For a quadratic model $f(x) = ax^2 + bx + c$, we construct a matrix $X$ containing **$x^2, x$, and 1**.  
- The **same matrix formula applies**:  

$$ f = X\Theta $$  

- This allows using the same algorithm for more complex models.

## Approach & Implementation

To breakdown the approach, we'll follow the 4 steps mentioned before.

### 1. Dataset

The dataset used in this implementation consists of data points with two main attributes for each car: **mileage** (`km`) and **price** (`price`). This dataset is stored in a CSV file, where each row represents one car. The mileage is the feature (independent variable), and the price is the target value (dependent variable).

In the code, the `get_data` function retrieves this data from the CSV file. It extracts the **mileage** values into the `x` array and the **price** values into the `y` array, which are used for training the model.

Mathematically, the dataset consists of pairs of input-output values $(x_i, y_i)$ for each car, where:

- $x_i$ is the mileage of the $\)-th car,
- $y_i$ is the price of the $\)-th car,

The task is to use the mileage values $x$ to predict the price $y$ using a linear model.

### 2. Model

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

### 3. Cost Function

The **cost function** measures the performance of the model by calculating the error between the predicted values and the true values. In this implementation, the **Mean Squared Error (MSE)** is used as the cost function, which is commonly used for regression tasks. The formula for MSE is:

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

### 4. Learning Algorithm

The **learning algorithm** in this case is **Gradient Descent**, an iterative optimization method used to minimize the cost function by adjusting the model parameters $\theta_0$ and $\theta_1$. The basic idea of gradient descent is to compute the gradients (partial derivatives) of the cost function with respect to each parameter, and then update the parameters in the direction that reduces the cost.

The gradient of the cost function with respect to $\theta_0$ and $\theta_1$ is computed as follows:

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

After calculating the gradients, **gradient descent** updates the parameters by subtracting a fraction of the gradient from the current values. The fraction is determined by the **learning rate** $\alpha$, which controls how large the updates are.

The parameters are updated using the following formulas:

$$
\theta_0 = \theta_0 - \alpha \cdot \frac{\partial J}{\partial \theta_0}
$$
$$
\theta_1 = \theta_1 - \alpha \cdot \frac{\partial J}{\partial \theta_1}
$$

The `gradient_descent` function implements these updates iteratively for a specified number of iterations, adjusting the parameters to reduce the cost function:

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

As the training progresses, the parameters $\theta_0$ and $\theta_1$ converge toward values that minimize the cost function, and the model's predictions improve.

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

