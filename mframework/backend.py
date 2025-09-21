import numpy as np

"""
A Backend contains a registry of primitive operations.
Calls to the backend are delegated to the appropriate Numpy/Cupy method implementations. 
"""

# Later: Union[np.ndarray, cp.ndarray]
type backend_dtype = np.ndarray

class Backend:
    def add(self, a: backend_dtype, b: backend_dtype) -> backend_dtype: raise NotImplementedError

class NumpyBackend(Backend):
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.add(a, b)

class CupyBackend(Backend):
    pass
