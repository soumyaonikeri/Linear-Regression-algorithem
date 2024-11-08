# Linear Regression from Scratch

This project implements a simple linear regression model from scratch in Python, without using any pre-built machine learning libraries for training the model. The linear regression algorithm is used for predicting a continuous output (y) based on a given input (X), following the equation `y = mx + b`, where `m` is the slope (weight) and `b` is the intercept (bias).

## Project Overview

### Steps:

1. **Training:**
   - Initialize the weights and bias to zero.
   - For each data point in the training set:
     - Predict the target using `y = mx + b`.
     - Calculate the error between predicted and actual values.
     - Use gradient descent to update the weight and bias to minimize the error.
   - Repeat the training steps for a set number of iterations (n).

2. **Testing:**
   - For a given test data point, use the trained model to predict the target using the equation `y = mx + b`.

### Implementation Details

The `LinearRegression` class is designed with the following components:
- **Attributes:** Learning rate (`lr`), number of iterations (`n_iters`), weights (`weights`), and bias (`bias`).
- **Methods:**
  - `fit(X, y)`: Trains the model by adjusting weights and bias using gradient descent.
  - `predict(X)`: Predicts the target for new data points using the trained model.
  - **Helper Function:**
    - `mse(y_test, predictions)`: Calculates the Mean Squared Error between actual and predicted values.

### Example Workflow

1. **Data Generation and Visualization:** We generate a dataset using `sklearn.datasets.make_regression`, split it into training and testing sets, and visualize it.
2. **Model Training:** Train the linear regression model using the training set.
3. **Evaluation:** Predict the output for the test set and compute Mean Squared Error.
4. **Visualization of Results:** Plot the training and testing data, along with the regression line predicted by the model.

## Requirements

- `numpy`
- `scikit-learn`
- `matplotlib`

To install these packages, run:
```bash
pip install numpy scikit-learn matplotlib
