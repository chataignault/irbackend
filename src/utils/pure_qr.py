from cla_utils.exercises10 import GMRES
from cla_utils.exercises9 import pure_QR
from cla_utils.exercises8 import hessenberg
from cla_utils.exercises3 import solve_U
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable

import sys
import os
sys.path.append(os.getcwd())


def generate_A(m: int) -> np.ndarray:
    return np.diag([2.] * m) + np.diag([-1.] * (m - 1), 1) + \
        np.diag([-1.] * (m - 1), -1)


def generate_hilbert(m) -> np.ndarray:
    return np.reshape(np.array(
        [1. / (i + j - 1) for i in range(1, m + 1) for j in range(1, m + 1)]), (m, -1))


def pure_QR_eigenvalues(A: np.ndarray, maxit: int = 10000, tol: float = 1e-5,
                        track: bool = False, shift: bool = False) -> np.ndarray:
    """
    Takes symmetric matrix as argument and returns an approximation of its eigenvalues
    """
    if A.shape != A.T.shape or not np.array_equal(A, A.T):
        raise ValueError("Matrix should be symmetrical")
    hessenberg(A)
    m = len(A)
    d = np.zeros(m)
    dtrack = []
    for k in range(m - 1, 0, -1):
        if track:
            A, *dtrack_ = pure_QR(A, maxit, tol, True, True, shift)
            dtrack.append(dtrack_)
        else:
            A = pure_QR(A, maxit, tol, True, False, shift)
        d[k] = A[-1, -1]
        A = A[:-1, :-1]
    # last eigenvalue
    d[0] = A[0, 0]

    if track:
        return d, np.concatenate(dtrack, axis=1)

    return d


def plot_convergence_qr_v_shifted(mat: str, m: int = 10):
    """
    Plot the evolution of the absolute value of the ultimate element on the first lower diagonal
    Compare the speed of the pure QR algorithm and the modified shifted algorithm
    """
    tol = 1e-13
    if mat == "tridiagonal example":
        A = generate_A(m)
    elif mat == "hilbert":
        A = generate_hilbert(m)
    else:
        raise ValueError(f"Matrix {mat} not implemeted")
    A, dtrack = pure_QR_eigenvalues(A[:, :], tol=tol, track=True, shift=False)
    if mat == "tridiagonal example":
        A = generate_A(m)
    elif mat == "hilbert":
        A = generate_hilbert(m)
    else:
        raise ValueError(f"Matrix {mat} not implemeted")
    A, dtrack_shifted = pure_QR_eigenvalues(A, tol=tol, track=True, shift=True)
    dtrack = dtrack[0]
    dtrack_shifted = dtrack_shifted[0]

    dtrack = np.abs(dtrack)
    dtrack_shifted = np.abs(dtrack_shifted)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(dtrack)), dtrack, label="normal", alpha=.7)
    ax.plot(
        range(
            len(dtrack_shifted)),
        dtrack_shifted,
        label="shifted",
        alpha=.7)
    ax.set_title("Last off-diagonal element modulus evolution")
    fig.suptitle("Pure QR algorithm for eigenvalue approximation : ")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("absolute value")
    ax.grid()
    plt.show()


def precond_example(A: np.ndarray) -> Callable:
    """
    Returns the precondition function taking the upper triangular elements of A
    """
    return lambda v: solve_U(np.triu(A), np.array([v]).T).T[0]


def plot_precond_convergence_comparison():
    nx = np.logspace(1, 8, num=10, base=2).astype(np.int64)
    nitx = []
    nitx_prec = []
    tol = 1e-8
    for n in nx:
        np.random.seed(37 * n)
        A = generate_A(n).astype(complex)
        b = np.random.randn(n).astype(complex) * 100
        _, nit = GMRES(A, b, maxit=10000, tol=tol, precond=None,
                       return_nits=True, relative_tolerance=True)
        A = generate_A(n).astype(complex)
        b = np.random.randn(n).astype(complex)
        _, nit_prec = GMRES(A, b, maxit=10000, tol=tol, precond=precond_example(
            A), return_nits=True, relative_tolerance=True)
        nitx.append(nit)
        nitx_prec.append(nit_prec)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nx, nitx, label="normal")
    ax.plot(nx, nitx_prec, label="preconditionned")
    ax.grid()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Iterations")
    ax.set_title(
        "Evolution of the number of iterations with increasing dimension")
    fig.suptitle("GMRES convergence")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    from matplotlib import pyplot as plt
    from cla_utils.exercises9 import pure_QR

    # test parameters
    tol = 1e-12
    m = 10

    A = generate_A(m)
    print(">>> The matrix A is : \n")
    print(A)

    D = pure_QR(A, 10000, tol, True)

    d = np.diag(D)

    print("\n\nThe eigenvalue approximation is : \n")
    print(d)
    # the eigenvalues are arranged in descending order

    d_true = np.array([2. + 2 * np.cos((k * np.pi) / (m + 1))
                      for k in range(1, m + 1)])

    err = d - d_true

    print("\n\n The error of approximation is :")
    print(err)

    plot_convergence_qr_v_shifted("tridiagonal example")
    plot_convergence_qr_v_shifted("hilbert")

    plot_precond_convergence_comparison()
