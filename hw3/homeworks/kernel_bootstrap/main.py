from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return pow(np.multiply.outer(x_i,x_j) + 1, d)
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma*pow(np.subtract.outer(x_i,x_j), 2))
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    X = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    alpha = np.linalg.solve(kernel_function(X, X, kernel_param) + _lambda * np.eye(x.shape[0]),y)
    return alpha
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    x_folds = np.split(x, num_folds)
    y_folds = np.split(y, num_folds)
    loss = []
    for val_fold in range(num_folds):
        print("Validating fold:", val_fold)
        x_train, x_val = np.delete(x_folds, val_fold, 0).flatten(), x_folds[val_fold]
        y_train, y_val = np.delete(y_folds, val_fold, 0).flatten(), y_folds[val_fold]
        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        #The line above trains a model on all the test folds (every fold except the validation fold) to get alpha,
        #now we need to use alpha to multiply with the x values in the validation fold to get an array of y_pred's
        #based on the alpha we trained to get, then get mean error/loss between y_pred's and the actual y values
        #in the valiation fold of the y array
        Z = (x_val - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        y_pred = np.dot(alpha, kernel_function((x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0), Z , kernel_param))
        loss.append(np.mean((y_pred - y_val)**2))
    return np.mean(loss)
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    result = (None, None)
    least_loss = float("inf")
    lambs = 10.0 ** (np.arange(-5, 0))
    dist_squared = [(a - b)**2 for idx, a in enumerate(x) for b in x[idx + 1:]]
    gams = [1 / (np.median(dist_squared)), 1 / (np.median(dist_squared))]
    for lamb in lambs:
        for gam in gams:
            loss = cross_validation(x, y, kernel_function=rbf_kernel, kernel_param=gam, _lambda=lamb, num_folds=num_folds)
            if loss < least_loss:
                result = (lamb, gam)
                least_loss = loss
    return result
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    result = (None, None)
    least_loss = float("inf")
    lambs = 10.0 ** (np.arange(-5, 0))
    ds = np.arange(5, 26)
    for lamb in lambs:
        for d in ds:
            loss = cross_validation(x, y, kernel_function=poly_kernel, kernel_param=d, _lambda=lamb, num_folds=num_folds)
            if loss < least_loss:
                result = (lamb, d)
                least_loss = loss
    return result
    #raise NotImplementedError("Your Code Goes Here")

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. Repeat A, B with x_300, y_300

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    x_data, y_data = x_1000, y_1000
    num_folds = 30 if (len(x_data) == 30) else 10

    best_poly_lamb, best_d = poly_param_search(x_data, y_data, num_folds)
    best_rbf_lamb, best_gam = rbf_param_search(x_data, y_data, num_folds)

    poly_alpha = train(x_data, y_data, kernel_function=poly_kernel, kernel_param=best_d, _lambda=best_poly_lamb)
    rbf_alpha = train(x_data, y_data, kernel_function=rbf_kernel, kernel_param=best_gam, _lambda=best_rbf_lamb)

    print("best_poly_lamb =", best_poly_lamb, "best_d =", best_d)
    print("best_rbf_lamb =", best_rbf_lamb, "best_gam =", best_gam)

    x = np.linspace(0, 1, num=100)

    x_data_mean = np.mean(x_data, axis=0)
    x_data_std = np.std(x_data, axis=0)
    x_normalized = (x - x_data_mean) / x_data_std
    poly_preds = np.dot(poly_alpha, poly_kernel((x_data - x_data_mean) / x_data_std, x_normalized, best_d))
    rbf_preds = np.dot(rbf_alpha, rbf_kernel((x_data - x_data_mean) / x_data_std, x_normalized, best_gam))

    plt.figure(0)
    plt.plot(x_data[np.argsort(x_data)], y_data[np.argsort(x_data)], "x", label='Original Data')
    plt.plot(x, f_true(x), label='True $f(x)$')
    plt.plot(x, poly_preds, label='$\widehat{f}_{poly}(x)$')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.ylim(-6, 6)
    plt.legend()
    plt.title('n=' + str(len(x_data)) + ' $\lambda$=' + str(best_poly_lamb) + ' d=' + str(best_d))
    plt.savefig('f_poly_n_' + str(len(x_data)))

    plt.figure(1)
    plt.plot(x_data[np.argsort(x_data)], y_data[np.argsort(x_data)], "x", label='Original Data')
    plt.plot(x, f_true(x), label='True $f(x)$')
    plt.plot(x, rbf_preds, label='$\widehat{f}_{rbf}(x)$')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.ylim(-6, 6)
    plt.legend()
    plt.title('n=' + str(len(x_data)) + ' $\lambda$=' + str(best_rbf_lamb) + ' $\gamma$=' + str(best_gam))
    plt.savefig('f_rbf_n_' + str(len(x_data)))
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
