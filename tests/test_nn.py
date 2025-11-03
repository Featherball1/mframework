import pytest
import numpy as np

import mframework.functional as F
from mframework.autograd.tensor import Tensor, Parameter
from mframework.nn.module import Module
from mframework.nn.modules.container import Sequential
from mframework.nn.modules.nn import Linear, ReLU, MSELoss, Softmax
from mframework.nn.modules.shape import Flatten


# ---------------------------------------------------------------------
# Basic Module mechanics
# ---------------------------------------------------------------------

def test_parameter_registration_via_setattr():
    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.w = Parameter(F.ones((2, 2))._data)
            self.b = Parameter(F.zeros((2,))._data)

        def forward(self, x):
            return x @ self.w + self.b

    m = MyModule()
    params = list(m.parameters())
    assert len(params) == 2
    assert all(isinstance(p, Parameter) for p in params)
    assert np.allclose(params[0]._data, np.ones((2, 2)))


def test_submodule_registration_and_parameter_recursion():
    class Sub(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter(F.ones((1,))._data)

        def forward(self, x): return x

    class Top(Module):
        def __init__(self):
            super().__init__()
            self.sub1 = Sub()
            self.sub2 = Sub()

        def forward(self, x):
            return self.sub1(x) + self.sub2(x)

    m = Top()
    params = list(m.parameters())
    assert len(params) == 2, "Should recursively include submodule params"


def test_train_eval_mode_recursive():
    class Leaf(Module):
        def forward(self, x): return x

    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = Leaf()

        def forward(self, x): return x

    p = Parent()
    assert p._training
    p.eval()
    assert not p._training and not p.child._training
    p.train()
    assert p._training and p.child._training


# ---------------------------------------------------------------------
# Sequential container
# ---------------------------------------------------------------------

def test_sequential_ordered_execution():
    lin1 = Linear(4, 3)
    relu = ReLU()
    lin2 = Linear(3, 2)
    model = Sequential(lin1, relu, lin2)

    x = F.ones((1, 4))
    out = model(x)
    assert isinstance(out, Tensor)
    # Check forward calls were chained properly
    assert out.shape == (1, 2)


def test_sequential_from_ordereddict():
    from collections import OrderedDict
    modules = OrderedDict([
        ('fc1', Linear(2, 2)),
        ('act', ReLU()),
        ('fc2', Linear(2, 1))
    ])
    seq = Sequential(modules)
    assert list(seq._submodules.keys()) == ['fc1', 'act', 'fc2']


# ---------------------------------------------------------------------
# Individual layers
# ---------------------------------------------------------------------

def test_linear_forward_shapes():
    m = Linear(4, 3)
    x = F.ones((2, 4))
    y = m(x)
    assert y.shape == (2, 3)


def test_relu_forward_nonnegative():
    relu = ReLU()
    x = Tensor(np.array([[-1.0, 0.5]]))
    y = relu(x)
    assert np.all(y._data >= 0)
    assert np.allclose(y._data, [[0.0, 0.5]])


def test_mseloss_forward_scalar_output():
    mse = MSELoss()
    pred = Tensor(np.array([[1.0, 2.0, 3.0]]))
    target = Tensor(np.array([[1.0, 0.0, 1.0]]))
    out = mse(pred, target)
    assert out.shape == (), "Loss should be scalar"
    assert np.isclose(out._data, np.mean(((pred._data - target._data) ** 2)))


def test_softmax_rows_sum_to_one():
    sm = Softmax()
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    y = sm(x)
    sums = y._data.sum(axis=-1)
    assert np.allclose(sums, 1.0)


def test_flatten_changes_shape_correctly():
    flat = Flatten()
    x = F.ones((2, 3, 4))
    y = flat(x)
    assert y.shape == (24,)
