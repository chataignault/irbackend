import numpy as np
from typing import Union, Tuple, Optional, Callable

from .householder import householder, compute_v_householder, householder_qr


def solve_U(U: np.ndarray, b: np.ndarray):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing
       the solution x_i

    """
    m, k = b.shape
    x = np.zeros((m, k), dtype=U.dtype)
    x[m - 1, :] = b[m - 1, :] / U[m - 1, m - 1]
    if m >= 2:
        for i in range(m - 2, -1, -1):
            x[i, :] = (b[i, :] - U[i, (i + 1) :] @ x[(i + 1) :, :]) / U[i, i]

    return x


def householder_solve(A: np.ndarray, b: np.ndarray):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m = len(A)
    U = np.concatenate([A, b], axis=1)
    householder(U, kmax=m)
    x = solve_U(U[:, :m], U[:, m:])

    return x


def householder_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    c = b.copy()
    m, k = A.shape
    if c.ndim == 1:
        c = np.reshape(c, (len(c), 1))
    A_hat = np.concatenate([A, c], axis=1)
    householder(A_hat, kmax=k + 1)
    R, c = A_hat[:k, :k], A_hat[:k, k:]
    x = solve_U(R, c)

    return x.T[0]


def Q1AQ1s(A: np.ndarray) -> np.ndarray:
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    v = compute_v_householder(A[:, 0])
    A[:, :] -= 2 * v @ (np.conj(v.T) @ A[:, :])
    A[:, :] -= 2 * (A[:, :] @ v) @ np.conj(v.T)

    return A


def hessenberg(A: np.ndarray):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = len(A)
    for k in range(m - 1):
        v = compute_v_householder(A[(k + 1) :, k])
        A[(k + 1) :, k:] -= 2 * v @ (np.conj(v.T) @ A[(k + 1) :, k:])
        A[:, (k + 1) :] -= 2 * (A[:, (k + 1) :] @ v) @ np.conj(v.T)


def hessenbergQ(A: np.ndarray) -> np.ndarray:
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array

    :return Q: an mxm numpy array
    """
    m = len(A)
    Q = np.eye(m)
    # Ap = np.concatenate([A, np.eye(m)], axis=1)
    for k in range(m - 2):
        v = compute_v_householder(A[(k + 1) :, k])
        Q[(k + 1) :, :] -= 2 * v @ (np.conj(v.T) @ Q[(k + 1) :, :])
        A[(k + 1) :, k:] -= 2 * v @ (np.conj(v.T) @ A[(k + 1) :, k:])
        A[:, (k + 1) :] -= 2 * (A[:, (k + 1) :] @ v) @ np.conj(v.T)
        # v = cla_utils.compute_v_householder(A[(k+1):, k])
        # Ap[(k+1):, k:] -= 2 * v @ (np.conj(v.T) @ Ap[(k+1):, k:])
        # A[(k+1):, k:] -= 2 * v @ (np.conj(v.T) @ A[(k+1):, k:])
        # A[:, (k+1):] -= 2 * (A[:, (k+1):] @ v) @ np.conj(v.T)
    # Q = Ap[:, -m:]
    return np.conj(Q.T)


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert m == n
    assert np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6
    _, V = np.linalg.eig(H)
    return V


def ev(A: np.ndarray) -> np.ndarray:
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    V = hessenberg_ev(A)

    return Q @ V


def pow_it(
    A: np.ndarray, x0: np.ndarray, tol: float, maxit: int, store_iterations=False
):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    n_iter = 0
    x = x0 / np.linalg.norm(x0)
    lambda0 = x @ A @ x
    while (n_iter < maxit) and (np.linalg.norm(A @ x - lambda0 * x) > tol):
        x = A @ x
        x = x / np.linalg.norm(x)
        lambda0 = x @ A @ x
        n_iter += 1

    return x, lambda0


def inverse_it(
    A: np.ndarray,
    x0: np.ndarray,
    mu: float,
    tol: float,
    maxit: int,
    store_iterations=False,
):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """
    n_iter = 0
    x = x0 / np.linalg.norm(x0)
    lambda0 = x @ A @ x
    I = np.eye(len(x))
    while (n_iter < maxit) and (np.linalg.norm(A @ x - lambda0 * x) > tol):
        x = householder_solve(A - mu * I, I) @ x
        x = x / np.linalg.norm(x)
        lambda0 = x @ A @ x
        n_iter += 1

    return x, lambda0


def rq_it(
    A: np.ndarray, x0: np.ndarray, tol: float, maxit: int, store_iterations=False
):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    n_iter = 0
    x = x0 / np.linalg.norm(x0)
    lambda0 = x @ A @ x
    I = np.eye(len(x))
    while (n_iter < maxit) and (np.linalg.norm(A @ x - lambda0 * x) > tol):
        x = householder_solve(A - lambda0 * I, I) @ x
        x = x / np.linalg.norm(x)
        lambda0 = x @ A @ x
        n_iter += 1

    return x, lambda0


def pure_QR_stop_cond_(A: np.ndarray, m: int) -> float:
    return np.linalg.norm(A[np.tril_indices(m, -1)]) / m**2


def pure_QR_stop_cond_trid_(A: np.ndarray, *args) -> float:
    return np.abs(A[-1, -2])


def pure_QR(
    A: np.ndarray,
    maxit: int,
    tol: float,
    trid: bool = False,
    track: bool = False,
    shift: Union[bool, complex] = False,
):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param track: yield -1 offset diagonal absolute value at each iteration

    :return Ak: the result
    """
    Ak = A.copy()
    m = len(A)
    n_iter = 0
    norm_cond = pure_QR_stop_cond_trid_ if trid else pure_QR_stop_cond_
    dtrack = []
    I = np.eye(m)
    while (n_iter < maxit) and (norm_cond(Ak, m) > tol):
        if track and (m > 1):
            dtrack.append(np.abs(Ak[-1, -2]))
        if (shift != False) and (n_iter > 1):
            x = Ak[:, -1]
            mu = (
                x @ Ak @ x / np.linalg.norm(x) ** 2
                if (isinstance(shift, bool))
                else shift
            )
            Ak = Ak - mu * I
        Q, R = householder_qr(Ak)
        Ak = R @ Q
        n_iter += 1

    if dtrack == []:
        return Ak

    return Ak, dtrack


def arnoldi(A: np.ndarray, b: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    Q = np.zeros((len(A), k + 1), dtype=complex)
    H = np.zeros((k + 1, k), dtype=complex)
    Q[:, 0] = b / np.linalg.norm(b)

    for i in range(k):
        v = A @ Q[:, i]
        H[: (i + 1), i] = np.conj(Q[:, : (i + 1)].T) @ v
        v -= Q[:, : (i + 1)] @ H[: (i + 1), i]
        h = np.linalg.norm(v)
        H[i + 1, i] = h
        Q[:, (i + 1)] = v / h

    return Q, H


def GMRES(
    A: np.ndarray,
    b: np.ndarray,
    maxit: int,
    tol: float,
    precond: Optional[Callable] = None,
    return_residual_norms=False,
    return_residuals=False,
    return_nits: bool = False,
    relative_tolerance: bool = False,
):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    res = np.linalg.norm(A @ b - b)
    if relative_tolerance:
        res = res / np.linalg.norm(b)
    if res < tol:
        return b, 0

    m = len(A)

    Q = np.zeros((m, 1), dtype=complex)
    H = np.array([])

    nb = np.linalg.norm(b)
    e = np.array([1, 0])
    Q[:, 0] = b / nb

    nits = 0

    while (nits < maxit) and (res > tol):
        Q = np.concatenate([Q, np.zeros((m, 1))], axis=1)
        if H.shape == (0,):
            H = np.zeros((2, 1), dtype=complex)
        else:
            H = np.concatenate([H, np.zeros((nits + 1, 1))], axis=1)
            H = np.concatenate([H, np.zeros((1, nits + 1))], axis=0)
        if precond:
            v = precond(A @ Q[:, nits])
        else:
            v = A @ Q[:, nits]
        H[: (nits + 1), nits] = np.conj(Q[:, : (nits + 1)].T) @ v
        v -= Q[:, : (nits + 1)] @ H[: (nits + 1), nits]
        h = np.linalg.norm(v)
        H[nits + 1, nits] = h
        Q[:, (nits + 1)] = v / h
        e = np.array(([1] + [0] * (nits + 1)))

        y = householder_ls(H, nb * e)
        res = np.linalg.norm(H @ y - nb * e)
        if relative_tolerance:
            res = res / nb
        nits += 1

    if res > tol:
        nits = -1

    x = Q[:, :-1] @ y

    if return_nits:
        return x, nits
    return x


def compute_pca(A: np.array):
    _, n = A.shape
    Q, R = householder_qr(A)
    return Q[:, :n], R[:n, :]


def get_principal_components(M: np.array):
    U, D, V = np.linalg.svd(M)
    return U, D, V
