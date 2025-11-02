import numpy as np
from typing import Union
from enum import Enum

"""
A Backend contains a registry of primitive operations.
Calls to the backend are delegated to the appropriate Numpy/Cupy method implementations. 

The API design here is flat to enable type hinting and autocomplete (instead of getattr/setattr tricks). 

As a rule of thumb - import numpy as np, import cupy as cp, should only appear in this file. 
"""

# Later: Union[np.ndarray, cp.ndarray]
type BackendArray  = Union[np.ndarray, "cp.ndarray"]

class BackendType(Enum):
    NUMPY = "numpy"
    CUPY = "cupy"

class Backend:
    def as_array(self, a) -> BackendArray:
        """
        Attempt to coerce `a` into a backend-suitable array type. 
        """
        raise NotImplementedError

    # Arithmetic operations
    def add(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def sub(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def mul(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def matmul(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def true_div(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def neg(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    
    # Shape operations
    def transpose(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    def reshape(self, a: BackendArray, newshape: tuple[int, ...]) -> BackendArray : raise NotImplementedError
    def flatten(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    def ndim(self, a: BackendArray) -> int : raise NotImplementedError
    def shape(self, a: BackendArray) -> tuple[int, ...] : raise NotImplementedError
    def expand_dims(self, a: BackendArray, axis: int | tuple[int, ...]) -> BackendArray: raise NotImplementedError
    def broadcast_to(self, a: BackendArray, shape: int | tuple[int, ...]) -> BackendArray: raise NotImplementedError

    # Reduction operations
    def sum(self, a: BackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BackendArray : raise NotImplementedError
    def prod(self, a: BackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BackendArray: raise NotImplementedError
    def mean(self, a: BackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BackendArray: raise NotImplementedError
    def maximum(self, a: BackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BackendArray : raise NotImplementedError
    def minimum(self, a: BackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BackendArray : raise NotImplementedError
    def max_eltwise(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def min_eltwise(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def argmax(self, a: BackendArray, axis: int | None = None, keepdims: bool = False) -> BackendArray: raise NotImplementedError
    def argmin(self, a: BackendArray, axis: int | None = None, keepdims: bool = False) -> BackendArray: raise NotImplementedError

    # BackendArray creation
    def ones(self, shape: tuple[int, ...]) -> BackendArray : raise NotImplementedError
    def zeros(self, shape: tuple[int, ...]) -> BackendArray : raise NotImplementedError
    def randn(self, *shape: int) -> BackendArray: raise NotImplementedError
    def where(self, condition: BackendArray, x: BackendArray, y: BackendArray) -> BackendArray : raise NotImplementedError
    def uniform(self, lb: float, ub: float, shape: tuple[int, ...]) -> BackendArray: raise NotImplementedError

    # Elementwise mathematical functions
    def exp(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    def log(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    def sqrt(self, a: BackendArray) -> BackendArray : raise NotImplementedError

    # Backend type
    @property
    def backend_type(self) -> BackendType: raise NotImplementedError

class NumpyBackend(Backend):
    def as_array(self, a) -> np.ndarray:
        try:
            return np.array(a, dtype=np.float32) if not isinstance(a, np.ndarray) else a
        except Exception:
            raise ValueError("Could not coerce `a` into a numpy array.")

    # Arithmetic operations
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.add(a, b)
    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.subtract(a, b)
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.multiply(a, b)
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.matmul(a, b)
    def true_div(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.true_divide(a, b)
    def neg(self, a: np.ndarray) -> np.ndarray: return np.negative(a)
    
    # Shape operations
    def transpose(self, a: np.ndarray) -> np.ndarray: return np.transpose(a)
    def reshape(self, a: np.ndarray, newshape: tuple[int, ...]) -> np.ndarray: return np.reshape(a, newshape)
    def flatten(self, a: np.ndarray) -> np.ndarray: return a.flatten()
    def ndim(self, a: np.ndarray) -> int: return a.ndim
    def shape(self, a: np.ndarray) -> tuple: return a.shape
    def expand_dims(self, a: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray: return np.expand_dims(a, axis)
    def broadcast_to(self, a: np.ndarray, shape: int | tuple[int, ...]) -> np.ndarray: return np.broadcast_to(a, shape)

    # Reduction operations
    def sum(self, a: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray: return np.sum(a, axis=axis, keepdims=keepdims)
    def prod(self, a: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray: return np.prod(a, axis=axis, keepdims=keepdims)
    def mean(self, a: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray: return np.mean(a, axis=axis, keepdims=keepdims)
    def maximum(self, a: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray: return np.max(a, axis=axis, keepdims=keepdims)
    def minimum(self, a: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray: return np.min(a, axis=axis, keepdims=keepdims)
    def max_eltwise(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.maximum(a, b)
    def min_eltwise(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.minimum(a, b)
    def argmax(self, a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray: return np.argmax(a, axis=axis, keepdims=keepdims)
    def argmin(self, a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray: return np.argmin(a, axis=axis, keepdims=keepdims)

    # BackendArray creation
    def ones(self, shape: tuple[int, ...]) -> np.ndarray: return np.ones(shape, dtype=np.float32)
    def zeros(self, shape: tuple[int, ...]) -> np.ndarray: return np.zeros(shape, dtype=np.float32)
    def randn(self, *shape: int) -> np.ndarray: return np.random.randn(*shape).astype(np.float32)
    def where(self, condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray: return np.where(condition, x, y)
    def uniform(self, lb: float, ub: float, shape: tuple[int, ...]) -> np.ndarray: return np.random.uniform(lb, ub, shape)

    # Elementwise mathematical functions
    def exp(self, a: np.ndarray) -> np.ndarray: return np.exp(a)
    def log(self, a: np.ndarray) -> np.ndarray: return np.log(a)
    def sqrt(self, a: np.ndarray) -> np.ndarray: return np.sqrt(a)

    @property
    def backend_type(self) -> BackendType: return BackendType.NUMPY

class CupyBackend(Backend):
    pass
