import numpy as np
import pytest

# Can test any backend
from mframework.backend import NumpyBackend
from mframework.autograd_utils import unbroadcast

backend = NumpyBackend()

"""
Testing for unbroadcasting utility function.
Unbroadcasting is a key autograd operation to reverse numpy-style broadcasting during backpropagation.
"""

@pytest.mark.parametrize("grad, shape, expected", [
    # No broadcasting (identity)
    (np.array([[1., 2.], [3., 4.]]), (2, 2), np.array([[1., 2.], [3., 4.]])),

    # Broadcast over columns (input had shape (2,1))
    (np.ones((2, 3)), (2, 1), np.array([[3.], [3.]])),

    # Broadcast over rows (input had shape (1,3))
    (np.ones((2, 3)), (1, 3), np.array([[2., 2., 2.]])),

    # Scalar input (shape = ())
    (np.ones((3, 2)), (), np.array(6.)),

    # Extra leading dimension in grad
    (np.ones((4, 2, 3)), (2, 3), np.ones((2, 3)) * 4),

    # Grad has fewer dims than input
    (np.array(2.), (2, 3), np.ones((2, 3)) * 2),

    # Mixed broadcasting (like (2,1,3) -> (2,4,3))
    (np.ones((2, 4, 3)), (2, 1, 3), np.ones((2, 1, 3)) * 4),
])

def test_unbroadcast(grad, shape, expected):
    result = unbroadcast(grad, shape, backend)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
