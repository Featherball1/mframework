from typing import Callable, Any
from dataclasses import dataclass

from mframework.backend import BackendType, Backend
from mframework import functional as F
from mframework.tensor import Tensor

"""
opinfo.py is part of the mframework testing infrastructure.

OpInfo defines metadata used for testing ops.
op_db is a list of OpInfo entries for all ops to be tested.

Adding an entry to op_db is effective to automatically including it in the test suite. 

OpInfo and op_db live in tests/_internal to avoid polluting mframework with test-related metadata. 
"""

@dataclass
class OpInfo:
    """
    Operator metadata for testing.
    Might later include:
        - Supports higher order gradients
        - Supports in place
        - dtypes (if we ever expand from float32)
    """
    name: str
    op: Callable
    sample_inputs: Callable[[Backend], list[tuple[Any, ...]]]
    supports_autograd: bool
    broadcasts: bool
    backends: list[BackendType]

"""
For the future - if this list gets large, create a folder
op_db/
    arithmetic ops
    function ops
    ...
so on,
and concatenate them here as 
op_db = arithmetic_ops + function_ops + ...
"""

op_db = [
    # Arithmetic ops
    OpInfo(
        name="add",
        op=F.add,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([1, 2, 3]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([4, 5, 6]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[1], [2]]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([10, 20, 30]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=True,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="sub",
        op=F.sub,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([5, 6, 7]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([1, 2, 3]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[1], [2]]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([10, 20, 30]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=True,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="mul",
        op=F.mul,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([1, 2, 3]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([4, 5, 6]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[1], [2]]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([10, 20, 30]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=True,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="div",
        op=F.div,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([10, 20, 30]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([2, 4, 5]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[1], [2]]), backend=backend, requires_grad=True),
                Tensor(backend.as_array([10, 20, 30]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=True,
        backends=[BackendType.NUMPY],
    ),

    # Function ops
    OpInfo(
        name="exp",
        op=F.exp,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([-1.0, 0.0, 1.0]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[0.5, -0.5], [1.0, -1.0]]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="log",
        op=F.log,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([1.0, 2.0, 10.0]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[0.5, 1.0], [2.0, 4.0]]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="relu",
        op=F.relu,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([-2.0, 0.0, 3.0]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[-1.0, 2.0], [0.0, 4.0]]), backend=backend, requires_grad=True),
            )
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),

    # Reduction ops
    OpInfo(
        name="sum",
        op=F.sum,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([[1.0, 2.0], [3.0, 4.0]]), backend=backend, requires_grad=True),
            ),
            (
                Tensor(backend.as_array([[[1.0, 2.0], [3.0, 4.0]]]), backend=backend, requires_grad=True),
            )
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),

    # Shape ops
    OpInfo(
        name="transpose",
        op=F.transpose,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([[1.0, 2.0], [3.0, 4.0]]), backend=backend, requires_grad=True),
            ),
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),
    OpInfo(
        name="reshape",
        op=F.reshape,
        sample_inputs=lambda backend: [
            (
                Tensor(backend.as_array([[1.0, 2.0], [3.0, 4.0]]), backend=backend, requires_grad=True),
                (4, 1),
            ),
            (
                Tensor(backend.as_array([1.0, 2.0, 3.0, 4.0]), backend=backend, requires_grad=True),
                (2, 2),
            )
        ],
        supports_autograd=True,
        broadcasts=False,
        backends=[BackendType.NUMPY],
    ),
]
