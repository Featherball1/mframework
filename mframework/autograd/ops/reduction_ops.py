from mframework.autograd.function import Function, Context
from mframework.dtypes import DType
from mframework.autograd.autograd_utils import (
    unbroadcast,
)


"""
Reduction operations (sum, mean, etc.)
"""


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(ctx.backend.shape(a), axis, keepdims)
        return ctx.backend.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a_shape, axis, keepdims = ctx.saved_for_backward
        backend = ctx.backend

        # If axis=None we have a full reduction and the result is a scalar
        if axis is None:
            # Re-expand to full shape note broadcast_to will raise if grad_out isn't at least 1D
            g = grad_out
            g = backend.reshape(g, (1,) * len(a_shape))
            g = backend.broadcast_to(g, a_shape)
            return (g,)

        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis

        g = grad_out

        # If reduction did not keep dims, we must reinsert dims of size 1
        if not keepdims:
            for ax in sorted(axes):
                g = backend.expand_dims(g, ax)

        # Now broadcast to input shape
        g = backend.broadcast_to(g, a_shape)

        return (g,)


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a.shape, axis, keepdims)
        return ctx.backend.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a_shape, axis, keepdims = ctx.saved_for_backward
        # divisor: number of elements of a contributing to each mean element
        if axis is None:
            divisor = int(ctx.backend.prod(a_shape))
        else:
            if isinstance(axis, int):
                axes = (axis,)
            else:
                axes = tuple(axis)
            divisor = ctx.backend.prod([a_shape[i] for i in axes])
        g = grad_out / divisor
        # Ensure reduced axes are present before unbroadcast
        if axis is not None and not keepdims:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                g = ctx.backend.expand_dims(g, ax)
        g = ctx.backend.broadcast_to(g, a_shape)
        return (unbroadcast(g, a_shape, ctx.backend),)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        backend = ctx.backend
        max_vals = backend.maximum(a, axis=axis, keepdims=True)
        mask = backend.where(a == max_vals,
            backend.ones(backend.shape(a)),
            backend.zeros(backend.shape(a))
        )
        ctx.save_for_backward(mask, backend.shape(a))
        return backend.maximum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        backend = ctx.backend
        mask, a_shape = ctx.saved_for_backward
        grad = mask * grad_out
        return (unbroadcast(grad, a_shape, backend),)


class Min(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        backend = ctx.backend
        min_vals = backend.minimum(a, axis=axis, keepdims=True)
        mask = backend.where(a == min_vals,
            backend.ones(backend.shape(a)),
            backend.zeros(backend.shape(a))
        )
        ctx.save_for_backward(mask, backend.shape(a))
        return backend.minimum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        backend = ctx.backend
        mask, a_shape = ctx.saved_for_backward
        grad = mask * grad_out
        return (unbroadcast(grad, a_shape, backend),)


class Conv2D(Function):
    @staticmethod
    def forward(ctx, X, W, b, stride=1, padding=0):
        backend = ctx.backend
        N, C_in, H, W_in = X.shape
        C_out, _, kH, kW = W.shape

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W_in + 2 * padding - kW) // stride + 1

        # (N, C_in*kH*kW, H_out*W_out)
        X_col = backend.stack([backend.im2col(X[n], kH, kW, stride, padding) for n in range(N)])

        # (C_out, C_in*kH*kW)
        W_col = backend.reshape(W, (C_out, -1))

        # (N, C_out, H_out*W_out)
        Y = backend.matmul(W_col, X_col)

        if b is not None:
            Y = Y + backend.reshape(b, (1, C_out, 1))

        Y = backend.reshape(Y, (N, C_out, H_out, W_out))

        ctx.save_for_backward(X, W)
        ctx.X_col = X_col
        ctx.b_exists = b is not None
        ctx.stride = stride
        ctx.padding = padding

        return Y

    @staticmethod
    def backward(ctx, grad_out):
        backend = ctx.backend
        X, W = ctx.saved_for_backward
        X_col = ctx.X_col
        N, C_in, H, W_in = X.shape
        C_out, _, kH, kW = W.shape
        _, _, H_out, W_out = grad_out.shape

        # (C_out, C_in*kH*kW)
        W_col = backend.reshape(W, (C_out, -1))
        dY = backend.reshape(grad_out, (N, C_out, H_out * W_out))

        # db: sum over batch and spatial dims
        db = backend.sum(dY, axis=(0, 2)) if ctx.b_exists else None

        # dW: (C_out, C_in*kH*kW)
        dW_col = backend.sum(
            backend.matmul(dY, backend.transpose(X_col, (0, 2, 1))),
            axis=0
        )
        dW = backend.reshape(dW_col, W.shape)

        # dX_col: (N, C_in*kH*kW, H_out*W_out)
        dX_col = backend.matmul(backend.transpose(W_col), dY)

        # col2im handles full batch at once
        dX = backend.col2im(dX_col, (N, C_in, H, W_in), kH, kW, ctx.stride, ctx.padding)

        if ctx.b_exists:
            return (dX, dW, db, None, None)
        else:
            return (dX, dW, None, None, None)


class MaxPool2D(Function):
    @staticmethod
    def forward(ctx, X, kernel_size, stride=1, padding=0):
        backend = ctx.backend
        N, C, H, W = X.shape
        kH, kW = kernel_size, kernel_size
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # (N, C, kH*kW, H_out*W_out)
        X_col = backend.stack([
            backend.stack([backend.im2col(X[n, c][None], kH, kW, stride, padding) for c in range(C)])
            for n in range(N)
        ])

        # (N, C, H_out*W_out)
        max_indices = backend.argmax(X_col, axis=2)
        Y = backend.max(X_col, axis=2).reshape(N, C, H_out, W_out)

        ctx.save_for_backward(X.shape, kernel_size, stride, padding, max_indices)
        return Y

    @staticmethod
    def backward(ctx, grad_out):
        X_shape, kernel_size, stride, padding, max_indices = ctx.saved_for_backward
        backend = ctx.backend
        N, C, H_out, W_out = grad_out.shape
        kH, kW = kernel_size, kernel_size
        H, W = X_shape[2], X_shape[3]
        hw = H_out * W_out

        # grad_col: (N, C, kH*kW, H_out*W_out)
        grad_col = backend.zeros((N, C, kH * kW, hw))
        grad_col[
            backend.arange(N)[:, None, None],
            backend.arange(C)[None, :, None],
             # (N, C, H_out*W_out)
            max_indices,
            backend.arange(hw)[None, None, :]
        ] = grad_out.reshape(N, C, hw)

        # Merge N and C into one batch dim for col2im
        # (N*C, kH*kW, H_out*W_out)
        grad_col_flat = grad_col.reshape(N * C, kH * kW, H_out * W_out)

        # col2im expects (batch, C*kH*kW, H_out*W_out)
        # with C=1 here since we flattened N and C together
        dX = backend.col2im(
            grad_col_flat,
            (N * C, 1, H, W),
            kH, kW,
            stride, padding
        ).reshape(N, C, H, W)

        return (dX, None, None, None)


class AvgPool2D:
    pass
