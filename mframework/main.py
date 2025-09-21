import numpy as np

from mframework.tensor import Tensor
from mframework.backend import NumpyBackend

if __name__ == "__main__":
    backend = NumpyBackend()
    tensor = Tensor(
        np.array([[1, 2], [3, 4]]),
        backend
    )
    tensor2 = Tensor(
        np.array([[1, 1], [1,1]]),
        backend
    )

    print((tensor + tensor2)._data)