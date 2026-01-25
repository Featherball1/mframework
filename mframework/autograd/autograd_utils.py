from typing import Tuple
import numpy as np

from mframework.autograd.backend import Backend, BackendArray

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

def gradcheck(fn, inputs, backend: Backend, eps=1e-4, tol=1e-3):
    """
    Finite difference gradient check.
    Used in the autograd testing strategy to verify local gradients match their finite difference approximation.

    For the purposes of autograd it is enough to consider gradcheck to be the "oracle" for local gradient testing.
    """
    # Forward pass
    out = fn(*inputs)
    out.backward()

    for x in inputs:
        if not getattr(x, "requires_grad", False):
            continue

        numerical_grad = np.zeros_like(x._data)
        it = np.nditer(x._data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = x._data[idx]
            x._data[idx] = old_val + eps
            out1 = fn(*inputs)._data.copy()
            x._data[idx] = old_val - eps
            out2 = fn(*inputs)._data.copy()
            x._data[idx] = old_val

            numerical_grad[idx] = backend.sum((out1 - out2) / (2 * eps))
            it.iternext()

        assert np.allclose(
            x.grad, numerical_grad, atol=tol
        ), f"Gradcheck failed for {fn.__name__} at {idx}"

def im2col(x, kH, kW, backend, stride=1, padding=0):
    C, H, W = x.shape

    if padding > 0:
        x = backend.pad(x, ((0,0),(padding,padding),(padding,padding)))

    H_p, W_p = x.shape[1:]
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    cols = []
    for i in range(out_h):
        for j in range(out_w):
            patch = x[
                :,
                i*stride:i*stride+kH,
                j*stride:j*stride+kW
            ]
            cols.append(patch.flatten())

    return backend.stack(cols, axis=1)

def col2im(cols, x_shape, backend, kH, kW, stride=1, padding=0):
    C, H, W = x_shape
    H_p, W_p = H + 2*padding, W + 2*padding
    x_padded = backend.zeros((C, H_p, W_p))

    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    col_idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = cols[:, col_idx].reshape(C, kH, kW)
            x_padded[
                :,
                i*stride:i*stride+kH,
                j*stride:j*stride+kW
            ] += patch
            col_idx += 1

    if padding > 0:
        return x_padded[:, padding:-padding, padding:-padding]
    return x_padded
