from typing import Tuple

from mframework.autograd.function import Function, Context
from mframework.autograd.backend import BackendArray
from mframework.autograd.autograd_utils import unbroadcast

"""
Autograd functions for basic functions of tensors. 
"""

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: BackendArray) -> BackendArray:
        ctx.save_for_backward(a)
        return ctx.backend.exp(a)

    @staticmethod
    def backward(ctx: Context, grad_out: BackendArray) -> Tuple[BackendArray,]:
        (a,) = ctx.saved_for_backward
        return (ctx.backend.mul(grad_out, ctx.backend.exp(a)),)

class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: BackendArray) -> BackendArray:
        ctx.save_for_backward(a)
        return ctx.backend.log(a)

    @staticmethod
    def backward(ctx: Context, grad_out: BackendArray) -> Tuple[BackendArray,]:
        (a,) = ctx.saved_for_backward
        return (ctx.backend.true_div(grad_out, a),)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: BackendArray) -> BackendArray:
        ctx.save_for_backward(a)
        return ctx.backend.max_eltwise(a, 0.0)

    @staticmethod
    def backward(ctx: Context, grad_out: BackendArray) -> Tuple[BackendArray,]:
        (a,) = ctx.saved_for_backward
        grad_a = ctx.backend.where(a > 0, ctx.backend.ones(ctx.backend.shape(a)), ctx.backend.zeros(ctx.backend.shape(a)))
        return (ctx.backend.mul(grad_out, grad_a),)

class MaxEltwise(Function):
    @staticmethod
    def forward(ctx: Context, a: BackendArray, b: BackendArray) -> BackendArray:
        ctx.save_for_backward(a, b)
        return ctx.backend.max_eltwise(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: BackendArray):
        a, b = ctx.saved_for_backward
        a_mask = ctx.backend.where(a > b, 1.0, 0.0)
        b_mask = ctx.backend.where(b > a, 1.0, 0.0)
        eq_mask = ctx.backend.where(a == b, 0.5, 0.0)
        a_mask = ctx.backend.add(a_mask, eq_mask)
        b_mask = ctx.backend.add(b_mask, eq_mask)
        return (
            unbroadcast(ctx.backend.mul(grad_out, a_mask), a.shape, ctx.backend),
            unbroadcast(ctx.backend.mul(grad_out, b_mask), b.shape, ctx.backend)
        )


class MinEltwise(Function):
    @staticmethod
    def forward(ctx: Context, a: BackendArray, b: BackendArray) -> BackendArray:
        ctx.save_for_backward(a, b)
        return ctx.backend.min_eltwise(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: BackendArray):
        a, b = ctx.saved_for_backward
        a_mask = ctx.backend.where(a < b, 1.0, 0.0)
        b_mask = ctx.backend.where(b < a, 1.0, 0.0)
        eq_mask = ctx.backend.where(a == b, 0.5, 0.0)
        a_mask = ctx.backend.add(a_mask, eq_mask)
        b_mask = ctx.backend.add(b_mask, eq_mask)
        return (
            unbroadcast(ctx.backend.mul(grad_out, a_mask), a.shape, ctx.backend),
            unbroadcast(ctx.backend.mul(grad_out, b_mask), b.shape, ctx.backend)
        )
