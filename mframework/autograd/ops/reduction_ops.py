from mframework.autograd.function import Function, Context
from mframework.autograd.autograd_utils import unbroadcast

"""
Reduction operations (sum, mean, etc.)
"""

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(ctx.backend.shape(a))
        return ctx.backend.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (a_shape,) = ctx.saved_for_backward
        return (unbroadcast(grad_out, a_shape, ctx.backend),)

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
            divisor = 1
            for ax in axes:
                divisor *= a_shape[ax]
        g = grad_out / divisor
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
