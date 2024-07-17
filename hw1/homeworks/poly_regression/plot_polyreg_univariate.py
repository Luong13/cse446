import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, reg_lambda=0)
    model.fit(X, y)
    model0 = PolynomialRegression(degree=d, reg_lambda=1)
    model0.fit(X, y)
    model1 = PolynomialRegression(degree=d, reg_lambda=0.001)
    model1.fit(X, y)
    model2 = PolynomialRegression(degree=d, reg_lambda=0.000001)
    model2.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    ypoints0 = model0.predict(xpoints)
    ypoints1 = model1.predict(xpoints)
    ypoints2 = model2.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d}")
    line0 = plt.plot(xpoints, ypoints0, "y-", label="$\lambda$ = 1")
    line1 = plt.plot(xpoints, ypoints1, "r-", label="$\lambda$ = 0.001")
    line2 = plt.plot(xpoints, ypoints2, "g-", label="$\lambda$ = 0.000001")
    line = plt.plot(xpoints, ypoints, "b-", label="$\lambda$ = 0")
    plt.legend(loc='lower center')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
