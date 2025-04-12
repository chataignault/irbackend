import numpy as np
from typing import Tuple

from .householder import compute_v_householder

def householder_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the householder decomposition of Q as :
    Q = I - WY^T
    implemented with functional paradigm as opposed to the in place implemetation in exercises3.py
    """
    n, p = A.shape
    W = np.zeros((n, p))
    Y = np.zeros((n, p))
    B = A.copy()

    for i in range(p):
        beta = 2.0
        v = compute_v_householder(B[i:, i])
        B[i:, i:] -= 2 * v @ (np.conj(v.T) @ B[i:, i:])
        if i > 0:
            v = np.concatenate([np.zeros((i, 1)), v])
        Y[:, i] = v.T
        W[:, i] = beta * (v - np.dot(W[:, :i], np.dot(Y[:, :i].T, v))).T

    return W, Y


def W_Y_from_beta_vx(beta: float, vx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the matrices such that Q* = I - W.Y.T from the set of householder vectors
    """
    W = np.zeros(vx.shape)
    p = W.shape[1]
    W[:, 0] = beta * vx[:, 0]
    if p > 1:
        for i in range(1, p):
            W[:, i] = beta * (vx[:, i] - W[:, :i] @ (vx[:, :i].T @ vx[:, i]))

    return W, vx
