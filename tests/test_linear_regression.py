import numpy as np
import unittest
from fast_linear_regression.linear_regression import FastLinearRegression

class TestFastLinearRegression(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([2, 3, 5, 7, 11])
        self.model = FastLinearRegression(method='ols')

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.weights)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_score(self):
        self.model.fit(self.X, self.y)
        score = self.model.score(self.X, self.y)
        self.assertGreaterEqual(score, 0)

if __name__ == '__main__':
    unittest.main()
