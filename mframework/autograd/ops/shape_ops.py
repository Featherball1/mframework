from typing import Tuple

from mframework.autograd.function import Function, Context

"""
Shape operations (transpose, reshape, etc.)
"""

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.shape)
        return ctx.backend.transpose(a)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (a_shape,) = ctx.saved_for_backward
        return ctx.backend.transpose(grad_out)

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a, newshape: Tuple[int,...]):
        ctx.save_for_backward(a.shape)
        return ctx.backend.reshape(a, newshape)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (a_shape,) = ctx.saved_for_backward
        return ctx.backend.reshape(grad_out, a_shape)

class Flatten(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.shape)
        return ctx.backend.flatten(a)

    @staticmethod
    def backward(ctx, grad_out):
        (a_shape,) = ctx.saved_for_backward
        raise ctx.backend.reshape(grad_out, a_shape)

class Gather(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError