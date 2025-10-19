from typing import Any

from mframework.autograd.backend import Backend

"""
Functions
---------

Inspired by Pytorch, every tensor operation is either
    - a primitive Function
    - or a composition of Functions. 

The role of a Function is to define how to compute the forward pass and gradients during backward. 

Some Functions need to cache information for backward computation.
The purpose of a Context is to provide every Function with a place to cache information needed for this. 

Contexts are necessary because Functions are never instantiated - their methods are all @staticmethod and cannot hold state.
The Context is a tool to allow Functions to hold state if there is a need to. 


Inputs x --------> call tensor._apply -------> call op.forward ------>(later) -------> call op.backward
                         ^                      ^                                          ^
                         |                      |                                          |
                         |                      |                                          |
                    create a ctx   saves intermediate calcs in ctx            retrieve context for usage in autograd
"""

class Context:
    def __init__(self, backend: Backend):
        self.saved_for_backward: tuple[Any, ...] = ()
        self.backend = backend

    def save_for_backward(self, *tensors) -> None:
        """Save tensors for the backward pass as a tuple, stored in self.saved_for_backward"""
        self.saved_for_backward = tensors

class Function:
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any: 
        """ Define a formula for differentiating the function in forward mode. """
        raise NotImplementedError
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Any) -> Any: 
        """ Define a formula for differentiating the function in backward mode. """
        raise NotImplementedError
    @staticmethod
    def setup_context(ctx: Context, inputs: tuple[Any, ...], output: Any) -> Any: 
        """ Define the initialisation of the computation context for the function. """
        raise NotImplementedError
