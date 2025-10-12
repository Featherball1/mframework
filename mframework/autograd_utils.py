from typing import Tuple
import numpy as np

from mframework.backend import Backend, BackendArray

def unbroadcast(grad: BackendArray, shape: Tuple[int, ...], backend: Backend) -> BackendArray:
    """
    Reduce `grad` to match `shape` (undo numpy-style broadcasting).

    Strategy:
      - If grad is scalar, return ones(shape) * grad.
      - Pad `shape` with leading ones to match grad.ndim.
      - For each axis i (0..ndim-1): if target_dim == 1 and grad_dim > 1, sum over axis i with keepdims=True.
      - After looping, reshape to `shape`.
    """
    grad = backend.as_array(grad)
    target_shape = tuple(shape)

    # scalar gradient
    if backend.ndim(grad) == 0:
        return backend.ones(target_shape) * grad

    gshape = list(backend.shape(grad))
    tshape = list(target_shape)

    # pad target shape with leading ones to match grad rank
    if len(gshape) > len(tshape):
        tshape = [1] * (len(gshape) - len(tshape)) + tshape
    elif len(tshape) > len(gshape):
        # grad has fewer dims than target: reshape grad with leading ones
        grad = backend.reshape(grad, (1,) * (len(tshape) - len(gshape)) + tuple(gshape))
        gshape = list(backend.shape(grad))

    # now gshape and tshape have same length
    for axis, (gdim, tdim) in enumerate(zip(backend.shape(grad), tshape)):
        if tdim == 1 and gdim != 1:
            # sum across this axis, keep dim so subsequent axes indices remain valid
            grad = backend.sum(grad, axis=axis, keepdims=True)

    # final reshape to exact target shape
    grad = backend.reshape(grad, tuple(target_shape))
    return grad
