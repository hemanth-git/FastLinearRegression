import numpy as np

class FastLinearRegression:
    def __init__(self, method='ols', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if self.method == 'ols':
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.method == 'gd':
            self.weights = np.zeros(X.shape[1])
            for _ in range(self.n_iterations):
                predictions = X @ self.weights
                errors = predictions - y
                gradient = X.T @ errors / len(y)
                self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights

    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
