from typing import Callable, Type, Tuple

import numpy as np

from mframework.autograd.backend import Backend, BackendArray, NumpyBackend
from mframework.autograd.function import Function, Context
from mframework.autograd.ops import *

DEFAULT_BACKEND = NumpyBackend()

"""
Tensor.

Two key methods:
    - _apply: apply a function to this tensor and create a closure for its local gradient
    - backward: run backpropagation from this tensor

Local gradients:
    - Each tensor resulting from an operation has a _local_grad closure that calls the relevant Function.backward
      and scatters gradients to its parents.

Backpropagation:
    - The backward method does a topological sort of the computation graph and calls each tensor's _local_grad in reverse order.
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
        self._local_grad: Callable | None = None
        self._grad: BackendArray = None

    def backward(self) -> None:
        if self._grad is None:
            self._grad = self._backend.ones(self._data.shape)

        topo_sorted: list["Tensor"] = []
        visited: set["Tensor"] = set()

        def build_topo(t: "Tensor"):
            if t not in visited:
                visited.add(t)
                if t._children is not None:
                    for child in t._children:
                        build_topo(child)
                topo_sorted.append(t)

        build_topo(self)
    
        for tensor in reversed(topo_sorted):
            if tensor._local_grad:
                tensor._local_grad()

    """
    Core internal class methods - wiring up ops and tensors. 
    """

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
        Instead of writing `t._grad += g` in `_local_grad`, the `_add_grad` function is a
        central entry point that allows us more detailed control (adding hooks, handling broadcasting, modifying gradients globally, etc). 
        If t._grad is None (ie not yet initialised), sets it to all zeros.  
        TODO:
            Handle hooks
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
        The returned Tensor has a _local_grad closure that calls cls.backward and scatters gradients.

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

        def _local_grad():
            if out._grad is None: return
            grads = f.backward(ctx, out._grad)
            if grads is None: return
            # Grads corresponds positionally to *args
            if not isinstance(grads, tuple):
                grads = (grads,)
            gi = 0
            for inp in args:
                if isinstance(inp, Tensor):
                    g = grads[gi]
                    if g is not None:
                        gi += 1
                        self._add_grad(inp, g)
                else:
                    # For a non-Tensor arg, consume on entry from the grads
                    gi += 1

        out._local_grad = _local_grad

        return out

    # Arithmetic operations
    def __add__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Add, self, other)
    def __sub__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Sub, self, other)
    def __mul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Mul, self, other)
    def __truediv__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Div, self, other)
    def __matmul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(MatMul, self, other)
    def __neg__(self) -> "Tensor":
        return self._apply(Neg, self)

    # Reduction operations
    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        return self._apply(Sum, self, axis=axis, keepdims=keepdims)

    # Shape operations
    def transpose(self) -> "Tensor":
        return self._apply(Transpose, self)
    def reshape(self, newshape: Tuple[int,...]) -> "Tensor":
        return self._apply(Reshape, self, newshape)
    def flatten(self) -> "Tensor":
        return self._apply(Flatten, self)

    # Basic mathematical functions
    def exp(self) -> "Tensor":
        return self._apply(Exp, self)
    def log(self) -> "Tensor":
        return self._apply(Log, self)
    def relu(self) -> "Tensor":
        return self._apply(ReLU, self)

    # Properties
    @property
    def T(self) -> "Tensor":
        return self._apply(Transpose, self)
    @property
    def shape(self) -> tuple:
        return self._data.shape

    """
    Helper methods.
    """

    def detach(self) -> "Tensor":
        return Tensor(self._data.copy(), self._backend, requires_grad=False)
    def detach_(self) -> "Tensor":
        self._requires_grad = False
        self._children = set()
        self._backward = lambda : None
        return self

    # Gradient hooks...

    def __repr__(self) -> str:
        return f"Tensor({self._data}, requires_grad={self._requires_grad})"

class Parameter(Tensor):
    """
    Taking inspiration from Pytorch, a parameter is a tensor that:
        - always has requires_grad = True
        - is recognised by module parameter lists
    """

    def __init__(self, data: BackendArray, backend: Backend) -> None:
        super().__init__(data, backend, requires_grad=True)

class Buffer(Tensor):
    """
    Taking inspiration from Pytorch, a buffer is a tensor that:
        - always has requires_grad = False
        - is recognised by module buffer lists
    """

    def __init__(self, data: BackendArray, backend: Backend) -> None:
        super().__init__(data, backend, requires_grad=False)
