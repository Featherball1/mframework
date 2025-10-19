from typing import Any
from mframework.autograd.function import Function, Context
from mframework.autograd.autograd_utils import unbroadcast

"""
Autograd Functions for arithmetic operations.
"""

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        """
        Note that the usage of ctx.backend.shape here is to ensure backend agnosticism.
        """
        ctx.save_for_backward(
            ctx.backend.shape(a),
            ctx.backend.shape(b)
        )
        return ctx.backend.add(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a_shape, b_shape = ctx.saved_for_backward
        return (unbroadcast(grad_out, a_shape, ctx.backend), unbroadcast(grad_out, b_shape, ctx.backend))

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return ctx.backend.sub(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a_shape, b_shape = ctx.saved_for_backward
        return (unbroadcast(grad_out, a_shape, ctx.backend), unbroadcast(grad_out, b_shape, ctx.backend))

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return ctx.backend.mul(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_for_backward
        a_shape, b_shape = ctx.backend.shape(a), ctx.backend.shape(b)
        return (
            unbroadcast(ctx.backend.mul(grad_out, b), a_shape, ctx.backend),
            unbroadcast(ctx.backend.mul(grad_out, a), b_shape, ctx.backend)
        )

class Div(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return ctx.backend.true_div(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b = ctx.saved_for_backward
        a_shape, b_shape = ctx.backend.shape(a), ctx.backend.shape(b)
        return (
            unbroadcast(ctx.backend.true_div(grad_out, b), a_shape, ctx.backend),
            unbroadcast(ctx.backend.neg(ctx.backend.true_div(ctx.backend.mul(grad_out, a), ctx.backend.mul(b, b))), b_shape, ctx.backend)
        )

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        # store both raw arrays and shapes
        ctx.save_for_backward(a, b, a.shape, b.shape)
        return ctx.backend.matmul(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out):
        a, b, a_shape, b_shape = ctx.saved_for_backward
        da = ctx.backend.matmul(grad_out, b.T)
        db = ctx.backend.matmul(a.T, grad_out)
        return (unbroadcast(da, a_shape, ctx.backend), unbroadcast(db, b_shape, ctx.backend))

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a):
        return ctx.backend.neg(a)

    @staticmethod
    def backward(ctx: Context, grad_out):
        return -grad_out
