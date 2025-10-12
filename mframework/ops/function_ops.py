from typing import Tuple

from mframework.function import Function, Context
from mframework.backend import Backend, BackendArray

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
