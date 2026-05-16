import numpy as np

"""
NUMPY BACKEND KERNELS

Some common ML operations (e.g. im2col, col2im) can be heavily optimised if we know the backend.
This file provides optimised implementations of such functions for each mframework backend. 
"""

def im2col_fast_np(x, kH, kW, stride=1, padding=0):
    # x: (C, H, W)
    if padding > 0:
        x = np.pad(x, ((0,0),(padding,padding),(padding,padding)))
    C, H, W = x.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    shape = (C, kH, kW, out_h, out_w)
    strides = (x.strides[0], x.strides[1], x.strides[2], 
               x.strides[1]*stride, x.strides[2]*stride)
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
    return patches.reshape(C*kH*kW, out_h*out_w)

def col2im_fast_np(X_col, x_shape, kH, kW, stride=1, padding=0):
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2*padding, W + 2*padding
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    X_pad = np.zeros((N, C, H_pad, W_pad))
    X_col_reshaped = X_col.reshape(N, C, kH, kW, H_out, W_out)

    # Precompute full index mesh — no Python loop
    i_idx = np.arange(kH)
    j_idx = np.arange(kW)
    h_idx = np.arange(H_out) * stride
    w_idx = np.arange(W_out) * stride

    # All destination indices: (kH, kW, H_out, W_out)
    h_dest = (i_idx[:, None, None, None] + h_idx[None, None, :, None])  # (kH, 1, H_out, 1)
    w_dest = (j_idx[None, :, None, None] + w_idx[None, None, None, :])  # (1, kW, 1, W_out)

    # Flatten kernel and spatial dims for a single add.at call
    # h_dest, w_dest broadcast to (kH, kW, H_out, W_out)
    h_dest = np.broadcast_to(h_dest, (kH, kW, H_out, W_out)).reshape(-1)
    w_dest = np.broadcast_to(w_dest, (kH, kW, H_out, W_out)).reshape(-1)

    # X_col_reshaped: (N, C, kH, kW, H_out, W_out) -> (N, C, kH*kW*H_out*W_out)
    vals = X_col_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, -1)

    np.add.at(X_pad, (slice(None), slice(None), h_dest, w_dest), vals)

    if padding > 0:
        return X_pad[:, :, padding:-padding, padding:-padding]
    return X_pad
