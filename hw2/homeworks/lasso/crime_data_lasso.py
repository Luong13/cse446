if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X = df_train.drop("ViolentCrimesPerPop", axis=1)
    y = df_train["ViolentCrimesPerPop"]

    X_test = df_test.drop("ViolentCrimesPerPop", axis=1)
    y_test = df_test["ViolentCrimesPerPop"]

    _lambda = max(2*np.sum(X.T * (y - np.mean(y)), axis=0))
    lambda_ratio = 2

    lambdas = []
    non_zeros = []
    mse_train = []
    mse_test = []

    non_zero = 0
    w_hat = None
    weights = []
    while _lambda > 0.001:
        #print("Training with lambda = ", _lambda)
        w_hat, b = train(X.values, y.values, _lambda, start_weight=w_hat)
        lambdas.append(_lambda)
        weights.append(w_hat.copy())
        non_zero = np.count_nonzero(w_hat)
        non_zeros.append(non_zero)

        y_train_preds = X.values.dot(w_hat) + b
        y_test_preds = X_test.values.dot(w_hat) + b

        mse_train.append(np.mean((y.values - y_train_preds)**2))
        mse_test.append(np.mean((y_test.values - y_test_preds)**2))

        """
        if _lambda > 20 and _lambda < 40:
            w_hat, b = train(X.values, y.values, _lambda=30.0, start_weight=w_hat)
            max_idx = np.argmax(w_hat)
            min_idx = np.argmin(w_hat)
            print('Max feature is ', X.columns[max_idx], ' at index ', max_idx)
            print('Min feature is ', X.columns[min_idx], ' at index ', min_idx)
        """
        _lambda /= lambda_ratio
    
    plt.figure(1)
    plt.plot(lambdas, non_zeros)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-Zeros')
    plt.savefig('A6c')

    parameters = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    indices = X.columns.get_indexer(parameters)
    plt.figure(2)
    for i in range(len(parameters)):
        reg_path = []
        for weight in weights:
            reg_path.append(weight[indices[i]])
        plt.plot(lambdas, reg_path, label=parameters[i])
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Regularization Paths')
    plt.legend()
    plt.savefig('A6d')

    plt.figure(3)
    plt.plot(lambdas, mse_test, label='Test')
    plt.plot(lambdas, mse_train, label='Train')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('A6e')

    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
