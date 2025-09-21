import numpy as np
from mframework.tensor import Tensor
from mframework.backend import NumpyBackend

def test_tensor_addition():
    backend = NumpyBackend()
    a = Tensor(np.array([[1, 2], [3, 4]]), backend)
    b = Tensor(np.array([[1, 1], [1, 1]]), backend)

    result = a + b
    expected = np.array([[2, 3], [4, 5]])

    assert np.array_equal(result._data, expected)
