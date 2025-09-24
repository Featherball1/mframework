import numpy as np
from mframework.tensor import Tensor
from mframework.backend import NumpyBackend

# test_tensor.py
import numpy as np
import pytest
from mframework.backend import NumpyBackend
from mframework.tensor import Tensor

""" 
Testing tensor promotion logic. 
"""

def test_add_tensor_same_backend():
    backend = NumpyBackend()
    a = Tensor(np.array([1, 2, 3]), backend)
    b = Tensor(np.array([4, 5, 6]), backend)
    c = a + b
    np.testing.assert_array_equal(c._data, np.array([5, 7, 9]))


def test_add_promotes_python_scalar():
    backend = NumpyBackend()
    a = Tensor(np.array([1, 2, 3]), backend)
    c = a + 3
    np.testing.assert_array_equal(c._data, np.array([4, 5, 6]))


def test_add_promotes_numpy_array():
    backend = NumpyBackend()
    a = Tensor(np.array([1, 2, 3]), backend)
    c = a + np.array([10, 20, 30])
    np.testing.assert_array_equal(c._data, np.array([11, 22, 33]))


def test_requires_grad_propagation():
    backend = NumpyBackend()
    a = Tensor(np.array([1, 2]), backend, requires_grad=True)
    b = Tensor(np.array([3, 4]), backend, requires_grad=False)
    c = a + b
    assert c._requires_grad is True


def test_backend_mismatch_raises():
    # Fake backend to simulate mismatch
    class DummyBackend(NumpyBackend):
        pass

    a = Tensor(np.array([1]), NumpyBackend())
    b = Tensor(np.array([2]), DummyBackend())
    with pytest.raises(ValueError, match="Backend of `other` is not the same as backend of `self`."):
        _ = a + b


def test_unpromotable_type_raises():
    backend = NumpyBackend()
    a = Tensor(np.array([1, 2, 3]), backend)
    with pytest.raises(ValueError, match="`other` is not a Tensor and cannot be promoted."):
        _ = a + dict()  # object() cannot be turned into a Tensor

"""
Tensor ops tests. 
"""

def test_tensor_addition():
    a = Tensor(np.array([[1, 2], [3, 4]]))
    b = Tensor(np.array([[1, 1], [1, 1]]))

    result = a + b
    expected = np.array([[2, 3], [4, 5]])

    assert np.array_equal(result._data, expected)

"""
Tensor properties tests.
"""

def test_tensor_shape():
    a = Tensor(np.array([[1, 2], [1, 2]]))
    assert a.shape == (2, 2)

def test_tensor_transpose():
    a = Tensor(np.array([[1, 2], [3, 4]])).T
    expected = Tensor(np.array([[1, 3], [2, 4]]))
    assert (a._data == expected._data).all()