import numpy as np

from .linear_algebra import householder_qr


def compute_pca(A: np.array):
    _, n = A.shape
    Q, R = householder_qr(A)
    return Q[:, :n], R[:n, :]


def get_principal_components(M: np.array):
    U, D, V = np.linalg.svd(M)
    return U, D, V
