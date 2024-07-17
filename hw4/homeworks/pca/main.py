from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    n,d = demean_data.shape
    mu = np.dot(demean_data.T, np.ones((n,1))) / n
    bruh = demean_data - mu.T
    breh = np.dot(uk, uk.T)
    return np.dot(bruh, breh) + mu.flatten()
    #raise NotImplementedError("Your Code Goes Here")


# @problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    This function has been implemented for you.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    tmp = demean_data - reconstruct_demean(uk, demean_data)
    # take the norm of each column, and then find the mean (of n values)
    res = np.mean(np.linalg.norm(tmp, axis=1) ** 2)
    return res


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of its covariance matrix.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,). Should be in descending order.
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    n,d = demean_data.shape
    one_vector = np.ones((n,1))
    mu = np.dot(demean_data.T, one_vector) / n
    Sigma = np.dot((demean_data - np.dot(one_vector, mu.T)).T, (demean_data - np.dot(one_vector, mu.T))) / n
    eig_vals, eig_vecs = np.linalg.eigh(Sigma)
    sorted_idx = np.argsort(eig_vals)[::-1]
    return (eig_vals[sorted_idx], eig_vecs[:,sorted_idx])
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.

    Note that you do not need to use reconstruction_error anywhere in the Winter 2023 iteration of this course.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    eig_vals, eig_vecs = calculate_eigen(x_tr)
    print(eig_vals[0], eig_vals[1], eig_vals[9], eig_vals[29], eig_vals[49])
    print(sum(eig_vals))

    idcs = []
    for digit in [2,6,7]:
        idcs.append(np.where(y_tr==digit)[0][0])
    ks = [5,15,40,100]
    plt.figure(0)
    fig,axes = plt.subplots(3,5)
    for row,idx in enumerate(idcs):
        axes[row,len(ks)].imshow(x_tr[idx].reshape(28,28))
        axes[row,len(ks)].set_title("Original")
        axes[row,len(ks)].axis("off")
        for col,k in enumerate(ks):
            rec = reconstruct_demean(eig_vecs[:,:k], x_tr)[idx]
            axes[row,col].imshow(rec.reshape(28,28))
            axes[row,col].set_title("k = {}".format(k))
            axes[row,col].axis("off")
    plt.savefig("A5c")
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
