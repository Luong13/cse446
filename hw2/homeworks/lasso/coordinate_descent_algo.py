from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    a = np.square(X)
    a = np.sum(a, axis=0)
    a = np.multiply(a,2.0)
    return a
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    b = np.mean(y - X.dot(weight))
    for k in range(len(weight)):
        a_k = a[k]
        c_k = 2 * np.sum(X[:,k] * (y - (b + X.dot(weight) - X[:,k].dot(weight[k]))), axis=0)
        if c_k < -_lambda:
            weight[k] = (c_k + _lambda) / a_k
        elif c_k > _lambda:
            weight[k] = (c_k - _lambda) / a_k
        else:
            weight[k] = 0

    return (weight, b)
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    lin_loss = np.sum(np.square(X.dot(weight) - y + bias))
    reg = _lambda * np.linalg.norm(weight,ord=1)
    return lin_loss + reg
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None
    while (old_w is None or not convergence_criterion(start_weight, old_w, convergence_delta)):
        old_w = np.copy(start_weight)
        start_weight, b = step(X, y, start_weight, a, _lambda)
    return start_weight, b
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    return np.max(np.absolute(weight - old_w)) < convergence_delta
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n, d, k = 500, 1000, 100
    w = np.zeros((d, ))
    for j in range(1, k+1):
        w[j-1] = j/k
    X = np.random.normal(size=(n, d))
    y = X.dot(w) + np.random.normal(size=(n,))

    _lambda = max(2*abs(np.sum(X*(y-np.mean(y))[:, None], axis=0)))
    lambda_ratio = 2

    lambdas = []
    non_zeros = []
    fdrs = []
    tdrs = []

    non_zero = 0
    w_hat = None
    while non_zero < d:
        #print("Training with lambda = ", _lambda)
        w_hat, b = train(X, y, _lambda, start_weight=w_hat)
        lambdas.append(_lambda)
        non_zero = np.count_nonzero(w_hat)
        non_zeros.append(non_zero)

        if non_zero > 0:
            fdrs.append(np.count_nonzero(w_hat[k:]) / non_zero)
        else:
            fdrs.append(0)
        tdrs.append(np.count_nonzero(w_hat[:k]) / k)

        _lambda /= lambda_ratio

    plt.figure(1)
    plt.plot(lambdas, non_zeros)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-Zeros')
    plt.savefig('A5a.png')

    plt.figure(2)
    plt.plot(fdrs, tdrs)
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('A5b.png')
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
