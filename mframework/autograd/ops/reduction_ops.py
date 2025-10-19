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
        # Should be backend agnostic
        return (unbroadcast(grad_out, a_shape, ctx.backend),)

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out):
        raise NotImplementedError

class Max(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out):
        raise NotImplementedError
    
class Min(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out):
        raise NotImplementedError
