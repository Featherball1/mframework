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

def test_cross_entropy_loss():
    """Test CrossEntropyLoss with various scenarios"""
    from mframework.nn.modules.nn import CrossEntropyLoss
    
    ce_loss = CrossEntropyLoss()
    
    # Test 1: Perfect predictions (loss should be near 0)
    logits = Tensor(np.array([[10.0, 0.0, 0.0],
                               [0.0, 10.0, 0.0],
                               [0.0, 0.0, 10.0]]))
    targets = np.array([0, 1, 2])
    loss = ce_loss(logits, targets)
    assert loss.shape == (), "Loss should be scalar"
    assert loss._data < 0.1, f"Perfect predictions should have low loss, got {loss._data}"
    
    # Test 2: Worst predictions (loss should be high)
    logits = Tensor(np.array([[0.0, 10.0, 10.0],
                               [10.0, 0.0, 10.0],
                               [10.0, 10.0, 0.0]]))
    targets = np.array([0, 1, 2])
    loss = ce_loss(logits, targets)
    assert loss._data > 5.0, f"Wrong predictions should have high loss, got {loss._data}"
    
    # Test 3: Known analytical result
    # For uniform logits [0,0,0] with 3 classes, loss should be -log(1/3) ≈ 1.0986
    logits = Tensor(np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]))
    targets = np.array([0, 1])
    loss = ce_loss(logits, targets)
    expected_loss = -np.log(1.0 / 3.0)
    assert np.isclose(loss._data, expected_loss, atol=1e-4), \
        f"Uniform logits loss should be {expected_loss:.4f}, got {loss._data:.4f}"
    
    # Test 4: Numerical stability with large logits
    logits = Tensor(np.array([[1000.0, 0.0, 0.0],
                               [0.0, 1000.0, 0.0]]))
    targets = np.array([0, 1])
    loss = ce_loss(logits, targets)
    assert not np.isnan(loss._data), "Loss should not be NaN with large logits"
    assert not np.isinf(loss._data), "Loss should not be inf with large logits"
    assert loss._data < 0.1, f"Large correct logits should give low loss, got {loss._data}"
    
    # Test 5: Single sample
    logits = Tensor(np.array([[2.0, 1.0, 0.1]]))
    targets = np.array([0])
    loss = ce_loss(logits, targets)
    assert loss.shape == (), "Loss should be scalar even for single sample"
    assert loss._data > 0, "Loss should be positive"
    
    # Test 6: Gradient check - loss should be differentiable
    logits = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
    targets = np.array([2])
    loss = ce_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None, "Gradients should flow back through loss"
    assert logits.grad.shape == logits.shape, "Gradient shape should match input shape"


def test_cross_entropy_loss_batch_handling():
    """Test that CrossEntropyLoss handles different batch sizes correctly"""
    from mframework.nn.modules.nn import CrossEntropyLoss
    
    ce_loss = CrossEntropyLoss()
    
    # Small batch
    logits_small = Tensor(np.random.randn(2, 5).astype(np.float32))
    targets_small = np.array([0, 3])
    loss_small = ce_loss(logits_small, targets_small)
    assert not np.isnan(loss_small._data)
    
    # Large batch
    logits_large = Tensor(np.random.randn(128, 10).astype(np.float32))
    targets_large = np.random.randint(0, 10, size=128)
    loss_large = ce_loss(logits_large, targets_large)
    assert not np.isnan(loss_large._data)
    assert loss_large.shape == ()

def test_cross_entropy_loss_with_tensor_targets():
    """Test CrossEntropyLoss when targets are Tensors instead of numpy arrays"""
    from mframework.nn.modules.nn import CrossEntropyLoss
    from mframework.dtypes import DType
    
    ce_loss = CrossEntropyLoss()
    
    logits = Tensor(np.array([[1.0, 2.0, 3.0],
                               [3.0, 2.0, 1.0]]))
    # Test with Tensor targets
    targets = Tensor(np.array([2, 0]), dtype=DType.INT64)
    loss = ce_loss(logits, targets)
    
    assert loss.shape == (), "Loss should be scalar"
    assert not np.isnan(loss._data), "Loss should not be NaN"
    assert loss._data > 0, "Loss should be positive"