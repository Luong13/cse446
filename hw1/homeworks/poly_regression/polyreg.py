"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean: float = None
        self.std: float = None
        #raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        result = X
        for i in range(2, degree + 1):
            result = np.c_[result, X**i]
        return result
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        n = len(X)
        X_ = self.polyfeatures(X, self.degree)

        self.mean = np.mean(X_, axis=0)
        self.std = np.std(X_, axis=0)

        X_ = (X_ - self.mean) / self.std
        X_ = np.c_[np.ones([n, 1]), X_]

        n, d = X_.shape
        
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)
        X_ = self.polyfeatures(X, self.degree)

        X_ = (X_ - self.mean) / self.std
        X_ = np.c_[np.ones([n, 1]), X_]

        return X_.dot(self.weight)
        #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    result = np.subtract(a,b)
    result = np.square(result)
    return np.mean(result)
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    model = PolynomialRegression(degree, reg_lambda)
    for i in range(1,n):
        x_instances = Xtrain[0:i+1]
        y_instances = Ytrain[0:i+1]

        model.fit(x_instances, y_instances)

        train_predictions = model.predict(x_instances)
        test_predictions = model.predict(Xtest)

        errorTrain[i] = mean_squared_error(y_instances, train_predictions)
        errorTest[i] = mean_squared_error(Ytest, test_predictions)

    return [errorTrain, errorTest]
    #raise NotImplementedError("Your Code Goes Here")
