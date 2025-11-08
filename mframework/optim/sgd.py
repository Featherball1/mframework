from typing import Iterable

from mframework.autograd.tensor import Parameter, Tensor
from mframework.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2
    ):
        super().__init__(params, lr=lr)
    
    def step(self):
        for param in self._params:
            """
            TODO: this is really bad.
            param._grad is saved as a BackendArray even though gradients of tensors should themselves be tensors.
            When we refactor the autograd system we'll make sure that we don't have to cast to tensor here!

            Another thing - we need to add promotion rules for floats to allow us to multiply by floats without casting to Tensor...
            """
            param._data = param._data - self.state_dict["lr"] * param._grad