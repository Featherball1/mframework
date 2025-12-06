from typing import Any

from mframework.autograd.backend import Backend, BackendArray

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

Note - functions don't know about tensors. They are just the next layer after a BackendArray responsible for bundling
together contexts, and forward and backward logic. 
"""

class Context:
    def __init__(self, backend: Backend):
        self.saved_for_backward: tuple[Any, ...] = ()
        self.backend = backend

    def save_for_backward(self, *args) -> None:
        """Save backend arrays for the backward pass as a tuple, stored in self.saved_for_backward"""
        self.saved_for_backward = args

class Function:
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> BackendArray: 
        """ Define a formula for differentiating the function in forward mode. """
        raise NotImplementedError
    @staticmethod
    def backward(ctx: Context, *grad_outputs: BackendArray) -> tuple[BackendArray, ...]: 
        """ Define a formula for differentiating the function in backward mode. """
        raise NotImplementedError
