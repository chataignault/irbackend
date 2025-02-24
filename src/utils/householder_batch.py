import numpy as np
from typing import List

from cla_utils import householder
from .householder_extension import W_Y_from_beta_vx


def householder_batch(A: np.ndarray, r: int):
    """
    Apply householder algorithm to slices of width r
    """
    _, p = A.shape
    n_batch = p // r + int(p % r != 0)
    batches = {
        "W": [],
        "Y": []
    }

    for b in range(n_batch):
        start = b*r
        end = min((b+1)*r, p) 
        beta, vx = householder(A[start:, start:end], decomposed=True)
        W, Y = W_Y_from_beta_vx(beta, vx)
        batches["W"].append(W.copy())
        batches["Y"].append(Y.copy())

        A[start:, end:] -= W @ (Y.T @ A[start:, end:]) 

    return batches

def reverse_householder_batch(R: np.ndarray, r: int, Wx: List[np.ndarray], Yx: List[np.ndarray]) -> None:
    """
    Apply the reverse operation to the output R of the householder algorithm in batches with batch size r
    provided the list of decomposed householder projectors
    The idea is to apply in reverse order the transpose of I - W.Y.T to the corresponding bottom right slice of the matrix R
    
    HOWEVER : the implemention works when applying each Q.T to the r first columns of the bottom right slice
              and then applying (as expected) Q to the columns on the right of that 
    """
    n, p = R.shape
    nb = len(Wx)
    for b in range(nb)[::-1]:
        start = r*b
        end = min((b+1)*r, p)
        R[start:, start:end] -= Wx[b] @ (Yx[b].T @ R[start:, start:end])
        # equivalently : R[start:, start:end] = (np.eye(n - start) - Wx[b] @ Yx[b].T) @ R[start:, start:end]
        # if end > start + 1:
        R[start:, end:] -= ((R[start:, end:].T @ Wx[b]) @ Yx[b].T ).T
            # equivalently : R[start:, end:] = (np.eye(n - start) - Wx[b] @ Yx[b].T).T @ R[start:, end:]
