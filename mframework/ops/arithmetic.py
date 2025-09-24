from mframework.function import Function, Context
import numpy as np

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return ctx.backend.add(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a_shape, b_shape = ctx.saved_for_backward
        # TODO: handle broadcasting here
        return grad_out, grad_out


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return ctx.backend.mul(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_for_backward
        return grad_out * b, grad_out * a

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a.shape)
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (a_shape,) = ctx.saved_for_backward
        # Should be backend agnostic
        return grad_out * np.ones(a_shape)

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.shape)
        return ctx.backend.transpose(a)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (a_shape,) = ctx.saved_for_backward
        return ctx.backend.transpose(grad_out)