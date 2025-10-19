from typing import Tuple
from mframework.autograd.tensor import Tensor

"""
There are two ways to call ops in mframework:
    - Via Tensor methods, e.g., a + b, a.exp(), a.sum()
    - Via functional API, e.g., mframework.add(a, b), mframeworkexp(a)
This file implements the functional API by calling the corresponding Tensor methods.
The functional API is then exposed in mframework/__init__.py to the end user. 
"""

# Arithmetic ops
def add(a: Tensor, b: Tensor) -> Tensor:
    return a + b

def sub(a: Tensor, b: Tensor) -> Tensor:
    return a - b

def mul(a: Tensor, b: Tensor) -> Tensor:
    return a * b

def div(a: Tensor, b: Tensor) -> Tensor:
    return a / b

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b

def neg(a: Tensor) -> Tensor:
    return -a


# Reduction ops
def sum(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return a.sum(axis=axis, keepdims=keepdims)


# Function ops
def exp(a: Tensor) -> Tensor:
    return a.exp()

def log(a: Tensor) -> Tensor:
    return a.log()

def relu(a: Tensor) -> Tensor:
    return a.relu()


# Shape ops
def transpose(a: Tensor) -> Tensor:
    return a.transpose()

def reshape(a: Tensor, newshape: Tuple[int, ...]) -> Tensor:
    return a.reshape(newshape)
