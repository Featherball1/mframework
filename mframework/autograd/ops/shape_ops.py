from typing import Tuple

from mframework.dtypes import DType
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
    def forward(ctx, input, indices, axis=0):
        axis = axis if axis >= 0 else input.ndim + axis
        indices = ctx.backend.as_array(indices, dtype=DType.INT64)
        ctx.save_for_backward(input.shape, indices, axis)
        return ctx.backend.take_along_axis(input, indices, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, indices, axis = ctx.saved_for_backward
        grad_input = ctx.backend.zeros(input_shape)

        idxs = ctx.backend.indices(ctx.backend.shape(grad_output), dtype=DType.INT64)

        idxs_axis = indices
        idxs_subs = []
        for d in range(len(input_shape)):
            if d == axis:
                idxs_subs.append(idxs_axis)
            else:
                idxs_subs.append(idxs[d])

        ctx.backend.add_at(grad_input, tuple(idxs_subs), grad_output)

        return grad_input, None, None
