# FastLinearRegression
Implementation of Fast Linear Regression for Large and small datasets

# FastLinearRegression

FastLinearRegression is a lightweight implementation of linear regression designed to provide fast performance for both Ordinary Least Squares (OLS) and Gradient Descent methods. Built using NumPy, this class is easy to integrate into your data analysis and machine learning projects.

## Features

- **Two fitting methods**: Choose between Ordinary Least Squares (OLS) and Gradient Descent (GD).
- **Simple interface**: Easy to fit and predict with just a few method calls.
- **Performance**: Optimized for speed while maintaining ease of use.

## Installation

Make sure you have Python and NumPy installed. You can install NumPy using pip:

```bash
pip install numpy
```
```python
import numpy as np
from fast_linear_regression import FastLinearRegression

# Sample dataset
X = np.random.rand(1000, 1)  # 1000 samples, 1 feature
y = 3 * X.squeeze() + np.random.randn(1000) * 0.5  # y = 3x + noise

# Create and fit the model
model = FastLinearRegression(method='ols')  # Use 'gd' for gradient descent
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate the model
print("R^2 score:", model.score(X, y))
```

## Tests
To run the tests
```bash
python -m unittest discover -s tests
```
