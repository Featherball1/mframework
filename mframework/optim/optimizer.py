from typing import Iterable

from mframework.autograd.tensor import Tensor, Parameter
from mframework.functional import zeros

class Optimizer:
    def __init__(self, params: Iterable[Parameter], **kwargs):
        self._params: Iterable[Parameter] = params
        self.state_dict: dict = kwargs

    def zero_grad(self):
        for param in self._params:
            """
            This is another reason why the ._grad attribute of a Tensor needs to also be a Tensor.
            We are zeroing it with functional.zeros, but then taking the BackendArray part with ._data!
            """
            param._grad = zeros(param._grad.shape)._data

    """
    For the future - note that some optimization algorithms (L-BFGS etc) require computing the model multiple times.
    So in the future we may want to allow a closure that computes the model output to be passed to step. 
    
    To optimize over Parameters, three main approaches:
        - For loop
        - For each
        - Fused
    for now - just a for loop over the parameters. 
    """
    def step(self):
        raise NotImplementedError
