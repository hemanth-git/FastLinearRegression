import numpy as np
from fast_linear_regression.linear_regression import FastLinearRegression

# Sample dataset
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + np.random.randn(1000) * 0.5

# Create and fit the model
model = FastLinearRegression(method='ols')
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate the model
print("R^2 score:", model.score(X, y))
