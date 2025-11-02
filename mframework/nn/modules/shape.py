from mframework.nn.module import Module
from mframework.autograd.tensor import Tensor

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten()
