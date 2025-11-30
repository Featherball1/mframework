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
        for p in self._params:
            if p.grad is not None:
                # raw backend arrays
                p._data = p._backend.sub(p._data, self.state_dict["lr"] * p.grad._data)
