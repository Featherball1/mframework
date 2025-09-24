from typing import Callable, Type

import numpy as np

from mframework.backend import Backend, BackendArray, NumpyBackend
from mframework.function import Function, Context
from mframework.ops.arithmetic import (
    Add, Mul,
    Sum,
    Transpose
)

DEFAULT_BACKEND = NumpyBackend()

"""
Tensor

"""

class Tensor:
    def __init__(
        self,
        data: BackendArray,
        backend: Backend | None = None,
        requires_grad: bool = False
    ) -> None:
        self._backend = backend if backend else DEFAULT_BACKEND
        self._data = self._backend.as_array(data)
        self._requires_grad = requires_grad
        self._children: set[Tensor] | None = None
        self._op_str: str | None = None
        self._backward: Callable | None = None
        self._grad: BackendArray = None
    
    def _promote_other(self, other: object) -> "Tensor":
        """
        Ensure `other` is a Tensor on the same backend as `self`.
        If `other` is not a Tensor, try to promote it.
        """
        if not isinstance(other, Tensor):
            try:
                return Tensor(other, self._backend, requires_grad=self._requires_grad)
            except Exception:
                raise ValueError("`other` is not a Tensor and cannot be promoted.")
        
        if type(other._backend) != type(self._backend):
            raise ValueError("Backend of `other` is not the same as backend of `self`.")
        
        return other
    
    def _add_grad(self, t: "Tensor", g: np.ndarray) -> None:
        """
        Accumulate gradient `g` into `t._grad`. 
        Instead of writing `t._grad += g` in `_backward`, the `_add_grad` function is a
        central entry point that allows us more detailed control (adding hooks, handling broadcasting, modifying gradients globally, etc). 
        If t._grad is None (ie not yet initialised), sets it to all zeros.  
        TODO:
            Handle hooks
            Safely handle broadcasting
        """
        if t._requires_grad == False: return
        if t._grad is None: t._grad = np.zeros_like(t._data, dtype=np.float32)
        # Hooks run here...
        # Broadcasting runs here...
        # Gradient accumulation step
        t._grad += g

    def _apply(self, f: Type[Function], *args, **kwargs):
        """
        Unwrap Tensor arguments into raw arrays, call forward(ctx, ...) and wrap output Tensor.
        The returned Tensor has a _backward closure that calls cls.backward and scatters gradients.

        Really, this method should belong to Function, but having it here avoids circular imports.
        """
        ctx = Context(self._backend)
        raw_args = [
            a._data if isinstance(a, Tensor) else a for a in args
        ]
        out_data = f.forward(ctx, *raw_args, **kwargs)

        requires_grad = any(
            # TODO: When we introduce the no_grad context manager, we need to amend this part
            isinstance(a, Tensor) and a._requires_grad for a in args
        )
        out = Tensor(out_data, requires_grad=requires_grad)
        out._children = {
            a
            for a in args if isinstance(a, Tensor)
        }
        out._op_str = f.__name__.upper()

        def _backward():
            if out._grad is None: return
            grads = f.backward(ctx, out._grad)
            if grads is None: return
            # Grads corresponds positionally to *args
            if not isinstance(grads, tuple):
                grads = (grads,)
            gi = 0
            for inp in args:
                if isinstance(inp, Tensor) and g is not None:
                    g = grads[gi]
                    gi += 1
                    self._add_grad(inp, g)
                else:
                    # For a non-Tensor arg, consume on entry from the grads
                    gi += 1

        out._backward = _backward

        return out

    def __add__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Add, self, other)
    def __mul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Mul, self, other)

    @property
    def T(self) -> "Tensor":
        return self._apply(Transpose, self)
    @property
    def shape(self) -> tuple:
        return self._data.shape


class Parameter(Tensor):
    """
    Taking inspiration from Pytorch, a parameter is a tensor that:
        - always has requires_grad = True
        - is recognised by module parameter lists
    """

    def __init__(self, data: BackendArray, backend: Backend) -> None:
        super().__init__(data, backend, requires_grad=True)
