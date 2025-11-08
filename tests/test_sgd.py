import pytest
import numpy as np

from mframework.optim.optimizer import Optimizer
from mframework.optim.sgd import SGD
from mframework.autograd.tensor import Tensor, Parameter


class DummyParam(Parameter):
    """Simple parameter mock for testing."""
    def __init__(self, data, grad=None):
        super().__init__(data)
        self._grad = grad


def test_optimizer_zero_grad_sets_grad_to_zero():
    p1 = DummyParam(np.array([1.0]), grad=np.array([5.0]))
    p2 = DummyParam(np.array([2.0]), grad=np.array([-3.0]))
    opt = Optimizer([p1, p2])

    opt.zero_grad()

    assert (p1._grad == 0).all()
    assert (p2._grad == 0).all()


def test_optimizer_step_not_implemented():
    p = DummyParam(np.array([1.0]), grad=np.array([0.5]))
    opt = Optimizer([p])
    with pytest.raises(NotImplementedError):
        opt.step()


def test_sgd_updates_parameters_correctly():
    data = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.1, 0.1])
    lr = 0.1

    p = DummyParam(data.copy(), grad=grad.copy())
    opt = SGD([p], lr=lr)

    # compute expected result manually
    expected = data - lr * grad

    opt.step()

    # SGD.step should have updated the parameter value
    np.testing.assert_allclose(p._data, expected, atol=1e-6)


def test_sgd_handles_multiple_parameters():
    p1 = DummyParam(np.array([1.0]), grad=np.array([1.0]))
    p2 = DummyParam(np.array([2.0]), grad=np.array([2.0]))
    lr = 0.5

    opt = SGD([p1, p2], lr=lr)
    expected_p1 = p1._data - lr * p1._grad
    expected_p2 = p2._data - lr * p2._grad

    opt.step()

    np.testing.assert_allclose(p1._data, expected_p1)
    np.testing.assert_allclose(p2._data, expected_p2)

# What should the behaviour be if there is no gradient?
# def test_sgd_does_nothing_if_no_grad():
#     p = DummyParam(np.array([1.0]), grad=None)
#     original = p._data.copy()
#     opt = SGD([p], lr=0.1)

#     # Should not crash or modify param if grad is None
#     opt.step()

#     np.testing.assert_allclose(p._data, original)
