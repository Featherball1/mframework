import numpy as np
from typing import Union

"""
A Backend contains a registry of primitive operations.
Calls to the backend are delegated to the appropriate Numpy/Cupy method implementations. 

The API design here is flat to enable type hinting and autocomplete (instead of getattr/setattr tricks). 
"""

# Later: Union[np.ndarray, cp.ndarray]
type BackendArray  = Union[np.ndarray, "cp.ndarray"]

class Backend:
    def as_array(self, a) -> BackendArray:
        """
        Attempt to coerce `a` into a backend-suitable array type. 
        """
        raise NotImplementedError

    def add(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def mul(self, a: BackendArray, b: BackendArray) -> BackendArray : raise NotImplementedError
    def transpose(self, a: BackendArray) -> BackendArray : raise NotImplementedError
    def shape(self, a: BackendArray) -> BackendArray : raise NotImplementedError


class NumpyBackend(Backend):
    def as_array(self, a) -> np.ndarray:
        try:
            return np.array(a, dtype=np.float32) if not isinstance(a, np.ndarray) else a
        except Exception:
            raise ValueError("Could not coerce `a` into a numpy array.")

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.add(a, b)
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.multiply(a, b)
    def transpose(self, a: np.ndarray) -> np.ndarray: return np.transpose(a)
    def shape(self, a: np.ndarray) -> tuple: return a.shape

class CupyBackend(Backend):
    pass
