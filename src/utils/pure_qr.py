import numpy as np
from typing import Callable

from . import hessenberg, pure_QR, solve_U


def pure_QR_eigenvalues(
    A: np.ndarray,
    maxit: int = 10000,
    tol: float = 1e-5,
    track: bool = False,
    shift: bool = False,
) -> np.ndarray:
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


def precond_example(A: np.ndarray) -> Callable:
    """
    Returns the precondition function taking the upper triangular elements of A
    """
    return lambda v: solve_U(np.triu(A), np.array([v]).T).T[0]
