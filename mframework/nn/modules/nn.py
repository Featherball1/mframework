from math import sqrt

from mframework.state import get_backend
from mframework.autograd.backend import Backend
from mframework.nn.module import Module
from mframework.autograd.tensor import Tensor, Parameter
import mframework.autograd.functional as F

# Layers

class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: Backend | None = None
    ):
        super().__init__()
        self._backend = backend if backend else get_backend()

        self._in_features: int = in_features
        self._out_features: int = out_features

        # Xavier initialization
        limit = sqrt(6 / (in_features + out_features))
        self.weight = Parameter(
            self._backend.uniform(-limit, limit, (out_features, in_features))
        )
        self.bias = Parameter(
            self._backend.zeros((out_features,))
        ) if bias else Tensor(self._backend.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias

# Activation

class ReLU(Module):
    def __init__(self, backend: Backend | None = None):
        super().__init__()
        self._backend = backend if backend else get_backend()
    
    def forward(self, x: Tensor) -> Tensor:
        return F.max_eltwise(
            x,
            Tensor(self._backend.zeros(x.shape))
        )

# Loss

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        # At the time I implemented this, I didn't have elementwise power implemented into ops
        return F.mean((x - target) * (x - target))

class Softmax(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x.exp() / x.exp().sum(axis=-1, keepdims=True)
