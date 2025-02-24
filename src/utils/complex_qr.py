from cla_utils import pure_QR
import numpy as np
import sys
import os
sys.path.append(os.getcwd())


def Am(m: int) -> np.ndarray:
    return np.diag([1.] * (m - 1), k=-1) + np.diag([-1.] * (m - 1), k=1)


def convergence_complex_shift(m: int):
    A = Am(m).astype(complex)
    mu = .285j
    R = pure_QR(A, maxit=10, tol=1e-3, trid=True)
    Rshifted = pure_QR(A, maxit=500, tol=1e-7, trid=True, shift=mu)
    return R[-1, -2], Rshifted[-1, -1]


if __name__ == "__main__":
    A = Am(10)
    print(">>> A : ")
    print(np.round(A, decimals=2))
    print()
    print(">>> After some iterations : ")
    R = pure_QR(A, maxit=10, tol=1e-2, trid=True)
    print(np.round(R, decimals=3))
    print()
    A = Am(10)
    print(">>> After QR algorithm with zero tolerance : ")
    Rmaxit = pure_QR(A, maxit=1000, tol=0., trid=True)
    print(np.round(Rmaxit, decimals=3))
    print()

    l, ls = convergence_complex_shift(10)
    print(l, ls)
