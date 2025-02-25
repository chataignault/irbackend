import pytest
import numpy as np
from numpy import random

from householder import householder


@pytest.mark.parametrize("m", [20, 40, 87])
def test_householder(m):
    random.seed(1878 * m)
    A = random.randn(m, m).astype(complex)
    random.seed(1879 * m)
    A += 1j * random.randn(m, m)
    A0 = 1.0 * A
    householder(A0)
    R = A0
    assert np.allclose(R, np.triu(R))  # check R is upper triangular
    assert np.norm(np.dot(np.conj(R.T), R) - np.dot(np.conj(A.T), A)) < 1.0e-6
