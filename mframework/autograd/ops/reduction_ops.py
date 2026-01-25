from mframework.autograd.function import Function, Context
from mframework.autograd.autograd_utils import (
    unbroadcast,
    im2col,
    col2im
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
        """
        X: (N, C_in, H, W)
        W: (C_out, C_in, kH, kW)
        b: (C_out,) or None
        """

        backend = ctx.backend
        N, C_in, H, W_in = X.shape
        C_out, _, kH, kW = W.shape

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W_in + 2 * padding - kW) // stride + 1

        Y = backend.zeros((N, C_out, H_out, W_out))

        W_col = backend.reshape(W, (C_out, -1))  # (C_out, C_in*kH*kW)

        X_cols = []
        for n in range(N):
            x_col = im2col(X[n], kH, kW, backend, stride, padding)
            X_cols.append(x_col)

            y_col = backend.matmul(W_col, x_col)

            if b is not None:
                y_col += backend.reshape(b, (-1, 1))

            Y[n] = backend.reshape(y_col, (C_out, H_out, W_out))

        # Save everything backward needs
        ctx.save_for_backward(
            X,
            W,
            X_cols,
            b is not None,
            stride,
            padding,
        )

        return Y

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: (N, C_out, H_out, W_out)
        """

        backend = ctx.backend
        X, W, X_cols, has_bias, stride, padding = ctx.saved_for_backward

        N, C_in, H, W_in = X.shape
        C_out, _, kH, kW = W.shape
        _, _, H_out, W_out = grad_out.shape

        W_col = backend.reshape(W, (C_out, -1))

        dX = backend.zeros(X.shape)
        dW_col = backend.zeros(W_col.shape)
        db = backend.zeros((C_out,)) if has_bias else None

        for n in range(N):
            dY_col = backend.reshape(
                grad_out[n],
                (C_out, H_out * W_out)
            )

            if has_bias:
                db += backend.sum(dY_col, axis=1)

            dW_col += backend.matmul(
                dY_col,
                backend.transpose(X_cols[n])
            )

            dX_col = backend.matmul(
                backend.transpose(W_col),
                dY_col
            )

            dX[n] = col2im(
                dX_col,
                (C_in, H, W_in),
                backend,
                kH,
                kW,
                stride,
                padding
            )

        dW = backend.reshape(dW_col, W.shape)

        if has_bias:
            return (dX, dW, db, None, None,)
        else:
            return (dX, dW, None, None, None,)
