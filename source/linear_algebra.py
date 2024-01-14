"""
completed from https://github.com/comp-lin-alg/comp-lin-alg-course
"""

import numpy as np

def compute_v_householder(x: np.ndarray) -> np.ndarray:
    """
    form the pivot vector of the householder algorithm
    x is the initial vector and s in the missing length
    """
    e1 = np.array([[int(k == 0)] for k in range(len(x))])
    r = np.linalg.norm(x)
    X = np.array([x]).T
    if x.dtype == complex:
        factor = 1. if x[0] == 0. else x[0] / np.absolute(x[0])
        v1 = X + factor * r * e1
        v2 = X - factor * r * e1
        if np.linalg.norm(v1) > np.linalg.norm(v2):
            v = v1
        else:
            v = v2
    else:
        sign = 1 if x[0] == 0. else np.sign(x[0])
        v = sign * r * e1 + X

    v = v / np.linalg.norm(v)
    return v

def householder(A: np.ndarray, kmax=None, decomposed: bool = False):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.
    """

    m, n = A.shape
    if kmax is None:
        kmax = n
    if decomposed:
        vx = np.zeros((m, kmax))
        beta = 2.

    for k in range(kmax):
        v = compute_v_householder(A[k:, k])
        A[k:, k:] -= 2. * v @ (np.conj(v.T) @ A[k:, k:])
        if decomposed:
            if k > 0:
                v = np.concatenate([np.zeros((k, 1)), v])
            vx[:, k] = v.T

    if decomposed:
        return beta, vx
    
def householder_qr(A: np.ndarray):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    U = np.concatenate([A, np.eye(m, dtype=A.dtype)], axis=1)
    householder(U, kmax=m)
    R = U[:, :n]
    Q = np.conj(U[:, n:].T)

    return Q, R
