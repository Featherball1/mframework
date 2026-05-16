from math import sqrt

from mframework.dtypes import DType
from mframework.state import get_backend
from mframework.autograd.backend import Backend
from mframework.nn.module import Module
from mframework.autograd.tensor import Tensor, Parameter
import mframework.functional as F

# Layers
class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self._in_features: int = in_features
        self._out_features: int = out_features

        # Xavier initialization
        limit = sqrt(6 / (in_features + out_features))
        self.weight = Parameter(
            F.uniform(-limit, limit, (out_features, in_features))._data
        )
        self.bias = Parameter(
            F.zeros((out_features,))._data
        ) if bias else Tensor(F.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        self._in_channels: int = in_channels
        self._out_channels: int = out_channels
        self._kernel_size: int = kernel_size
        self._stride: int = stride
        self._padding: int = padding

        # Xavier initialization
        limit = sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.weight = Parameter(
            F.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))._data
        )
        self.bias = Parameter(
            F.zeros((out_channels,))._data
        ) if bias else Tensor(F.zeros((out_channels,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv_2d(
            x,
            self.weight,
            b=self.bias if isinstance(self.bias, Parameter) else None,
            stride=self._stride,
            padding=self._padding
        )


class MaxPool2D(Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self._kernel_size: int = kernel_size
        self._stride: int = stride
        self._padding: int = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.maxpool_2d(
            x,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding
        )


# Activation
class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.max_eltwise(
            x,
            F.zeros(x.shape)
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


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: "Tensor", target: "Tensor") -> "Tensor":
        max_logits = logits.max(axis=1, keepdims=True).detach()
        logits_shifted = logits - max_logits 
        
        exp_logits = logits_shifted.exp()
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        log_sum = sum_exp.log()
        log_probs = logits_shifted - log_sum

        indices = target.reshape((-1, 1))
        chosen = log_probs.gather(indices, axis=1)
        
        loss = (-chosen).mean()
        return loss
