from dataclasses import dataclass
from collections import deque
from typing import Type, Tuple

from mframework.dtypes import DType
import mframework.functional as F
from mframework.state import get_backend
from mframework.autograd.backend import Backend, BackendArray
from mframework.autograd.function import Function, Context
from mframework.autograd.ops import *


"""
Autograd
--------

Forward pass:
    x ----> f(x) -----> h(x, y) = z = output
                            ^
                            |
    y ----> g(y) ------------

As the forward pass executes we dynamically build a graph to do backward with

    grad_L <---- grad_H <---- grad_F <---- grad_x (leaf)
                    ^
                    |
                    --------- grad_G <---- grad_y (leaf)

The arrows are edges which contain additional metadata about how the node fits into backprop.
In particular, they contain the parent tensor and the slot of the function that it fits into. 

    grad_L <- (edge: parent H, slot 0) - grad_H <- (edge: parent F, slot 1) - x (leaf)
                                            ^
                                            |
                                            ------ (edge: parent G, slot 2) - y (leaf)

When it is time to do backward, we run a dfs on the resulting graph to identify 
which nodes are participating and what their dependencies are.
This gives a table like

node     |      receives gradients from     |     in_degree
----------------------------------------------------------------
grad_L   |               H                  |             1
grad_H   |           G       F              |             2
grad_F   |               x                  |             1
grad_G   |               y                  |             1
grad_x   |                                  |             0
grad_y   |                                  |             0

During backward, the root node receives an implicit upstream gradient (typically ones), 
so it is treated as if it had a single incoming gradient source.

With the table prepared and root node ready we can execute backward. Create a ReadyQueue, which keeps
track of nodes with in-degree zero. Then repeatedly:
    1) Pop a ready node from the queue (order does not matter - can even be parallelised)
    2) Compute the local gradient at the ready node
    3) Send gradients to parents
    4) Decrement in_degree counters for each parents
    5) Enqueue parents whose dependency counter hits zero
"""


__all__ = [
    "Tensor", "Parameter", "Buffer",
    "backward",
]


"""
Autograd graph definitions and tensor definitions. 
"""


@dataclass(slots=True)
class Node:
    """
    A Node contains the data needed during the autograd backward pass. 
    """
    grad_buffer: BackendArray | None
    op: Type[Function]  # contains context and backward op
    ctx: Context
    parents: list["Edge"]
    in_degree: int | None


@dataclass(frozen=True, slots=True)
class Edge:
    """
    Represents a particular input of a function / an edge in the graph. 
    We store Tensors here because we can recover the ._node from the Tensor. 
    """
    destination: "Tensor"
    grad_slot: int


class Tensor:
    __slots__ = (
        "_backend",
        "_dtype",
        "_data",
        "_requires_grad",
        "_node",
        "_grad"
    )

    def __init__(
        self,
        data: BackendArray,
        backend: Backend | None = None,
        requires_grad: bool = False,
        dtype: DType = DType.FLOAT32,
    ) -> None:
        # Data
        self._backend = backend if backend else get_backend()
        self._dtype = dtype
        self._data = self._backend.as_array(data, dtype=dtype)

        # Data for autograd
        self._requires_grad = requires_grad
        self._node: Node | None = None  # For function (autograd graph) nodes
        self._grad: "Tensor | None" = None  # Leaf gradient accumulator

    def backward(self):
        self._grad = backward(self)
    
    def _apply(self, f: Type[Function], *args, **kwargs):
        return _apply(self, f, *args, **kwargs)

    def _promote_other(self, other: object) -> "Tensor":
        """
        Ensure `other` is a Tensor on the same backend as `self`.
        If `other` is not a Tensor, try to promote it.
        """
        if not isinstance(other, Tensor):
            try:
                return Tensor(self._backend.as_array(other), self._backend, requires_grad=self._requires_grad)
            except Exception:
                raise ValueError("`other` is not a Tensor and cannot be promoted.")
        
        if type(other._backend) != type(self._backend):
            raise ValueError("Backend of `other` is not the same as backend of `self`.")
        
        return other

    # Arithmetic operations
    def __add__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Add, self, other)
    def __sub__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Sub, self, other)
    def __mul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Mul, self, other)
    def __rmul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Mul, other, self)
    def __truediv__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(Div, self, other)
    def __matmul__(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(MatMul, self, other)
    def __neg__(self) -> "Tensor":
        return self._apply(Neg, self)

    # Reduction operations
    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        return self._apply(Sum, self, axis=axis, keepdims=keepdims)
    def mean(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        return self._apply(Mean, self, axis=axis, keepdims=keepdims)
    def max(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        return self._apply(Max, self, axis=axis, keepdims=keepdims)
    def min(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        return self._apply(Min, self, axis=axis, keepdims=keepdims)

    # Shape operations
    def transpose(self) -> "Tensor":
        return self._apply(Transpose, self)
    def reshape(self, newshape: Tuple[int,...]) -> "Tensor":
        return self._apply(Reshape, self, newshape)
    def flatten(self) -> "Tensor":
        return self._apply(Flatten, self)
    def gather(self, indices: "Tensor", axis: int = -1) -> "Tensor":
        return self._apply(Gather, self, indices, axis=axis)

    # Basic mathematical functions
    def exp(self) -> "Tensor":
        return self._apply(Exp, self)
    def log(self) -> "Tensor":
        return self._apply(Log, self)
    def relu(self) -> "Tensor":
        return self._apply(ReLU, self)
    def max_eltwise(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(MaxEltwise, self, other)
    def min_eltwise(self, other: object) -> "Tensor":
        other = self._promote_other(other)
        return self._apply(MinEltwise, self, other)
    
    def detach(self) -> "Tensor":
        """
        Returns a new Tensor that shares the same data but is detached from the computation graph.
        The returned tensor will have requires_grad=False and no _node.
        """
        return Tensor(
            data=self._data,
            backend=self._backend,
            requires_grad=False,
            dtype=self._dtype
        )
    
    def detach_(self):
        self._node = None
        self._grad = None

    @property
    def T(self) -> "Tensor":
        return self._apply(Transpose, self)
    @property
    def shape(self) -> tuple:
        return self._data.shape
    @property
    def grad(self) -> "Tensor | None":
        # If this tensor is a leaf, return the leaf accumulator.
        if self._node is None:
            return self._grad
        # Non-leaf: gradient accumulated on its node (may be None)
        return Tensor(self._node.grad_buffer, backend=self._backend, requires_grad=False)
    @property
    def data(self) -> BackendArray:
        return self._data
    @property
    def ndim(self) -> int:
        return self._data.ndim
    @property
    def item(self):
        return float(self.data)


    def __str__(self) -> str:
        return f"""{self.__class__.__name__}(data={self._data}, requires_grad={self._requires_grad}, backend={self._backend})"""

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(data={self._data}, requires_grad={self._requires_grad}, backend={self._backend})"""


class Parameter(Tensor):
    """
    A parameter is a tensor that:
        - always has requires_grad = True
        - is recognised by module parameter lists
    """

    def __init__(self, data: BackendArray, backend: Backend | None = None) -> None:
        super().__init__(data, backend, requires_grad=True)


class Buffer(Tensor):
    """
    A buffer is a tensor that:
        - always has requires_grad = False
        - is recognised by module buffer lists
    """

    def __init__(self, data: BackendArray, backend: Backend | None = None) -> None:
        super().__init__(data, backend, requires_grad=False)


"""
Autograd engine logic.
"""

"""
Forward pass. 
"""

def _apply(t: Tensor, f: Type[Function], *args, **kwargs):
    """
    Apply a Function to a tensor.
    Build the computation graph eagerly. 
    """

    ctx = Context(t._backend)
    # We have to unwrap the args to get the raw data, else we create an infinite recursion
    # due to the way that functions are currently implemented
    raw_args = [
        a._data if isinstance(a, Tensor) else a for a in args
    ]
    out_data: BackendArray = f.forward(ctx, *raw_args, **kwargs)

    # determine requires_grad: True if any tensor arg requires grad
    requires_grad = any(isinstance(a, Tensor) and a._requires_grad for a in args)
    out = Tensor(out_data, requires_grad=requires_grad)

    # If requires_grad, add to the graph for backprop
    if requires_grad:
        parents: list[Edge] = []
        for idx, a in enumerate(args):
            if isinstance(a, Tensor) and a._requires_grad:
                if a._requires_grad:
                    parents.append(Edge(
                        a,
                        idx
                    ))
        out._node = Node(
            grad_buffer = None,
            op = f,
            ctx = ctx,
            parents = parents,
            # Placeholder - will be computed just before doing backward as some nodes may not participate
            in_degree = None
        )

    return out

"""
Backward pass. 
"""

class _ReadyQueue:
    """
    Work queue for autograd nodes whose in-degree has reached zero.
    """
    def __init__(self):
        self._queue = deque()

    def push(self, node: Node):
        self._queue.append(node)

    def pop(self) -> Node:
        return self._queue.popleft()
    
    def __len__(self) -> int:
        return len(self._queue)
    

def _collect_graph(root: Node) -> list[Node]:
    """Collect reachable nodes via DFS and return list."""
    stack = [root]
    visited = set()
    nodes = []
    while stack:
        n = stack.pop()
        if id(n) in visited:
            continue
        visited.add(id(n))
        nodes.append(n)
        for e in n.parents:
            p = e.destination
            if p._node is not None:
                stack.append(p._node)
    return nodes


def _compute_indegrees(root: Node, ready: _ReadyQueue):
    nodes = _collect_graph(root)
    # initialize
    for n in nodes:
        n.in_degree = 0
    # each node's in_degree is number of children that will consume its grad
    # To compute this, iterate parents and increment the parent's in_degree
    for n in nodes:
        for e in n.parents:
            parent = e.destination
            if parent._node is not None:
                parent._node.in_degree += 1
    # nodes with in_degree == 0 are ready
    for n in nodes:
        if n.in_degree == 0:
            ready.push(n)


def _backward(root_node: Node) -> BackendArray:
    """Iterative backward engine. root_node.grad must be set to initial gradient Tensor."""
    queue = _ReadyQueue()
    _compute_indegrees(root_node, queue)

    while len(queue) > 0:
        node: Node = queue.pop()

        if node.grad_buffer is None:
            # Nothing to do
            continue

        grad_buffer: BackendArray = node.grad_buffer

        # Call op.backward to obtain gradients for parents
        local_grads: list[BackendArray] = node.op.backward(node.ctx, grad_buffer)

        # Iterate through edges and distribute the local grads
        for edge in node.parents:
            # Use the grad_slot recorded on the edge to select the correct gradient
            try:
                g = local_grads[edge.grad_slot]
            except IndexError:
                raise RuntimeError(
                    f"Backward for {node.op.__name__} returned {len(local_grads)} grads, "
                    f"but expected an entry at slot {edge.grad_slot} (parents: {len(node.parents)})."
                )

            parent_tensor: Tensor = edge.destination

            if parent_tensor._node is None:
                # parent is a leaf: accumulate into the parent_tensor._grad accumulator
                g_wrapped = Tensor(g, backend=parent_tensor._backend, requires_grad=False)
                if parent_tensor._grad is None:
                    parent_tensor._grad = g_wrapped
                else:
                    parent_tensor._grad = parent_tensor._grad + g_wrapped

            else:
                # parent has a Node: accumulate into the parent.node.grad_buffer
                parent_node = parent_tensor._node
                if parent_node.grad_buffer is None:
                    parent_node.grad_buffer = g
                else:
                    parent_node.grad_buffer = parent_node.grad_buffer + g

                # decrement in_degree and push if ready
                parent_node.in_degree -= 1
                if parent_node.in_degree == 0:
                    queue.push(parent_node)

        # Prune the graph
        node.parents = []
        node.grad_buffer = None


def backward(t: Tensor) -> Tensor:
    if t._node is None:
        raise RuntimeError("Tensor not attached to graph.")
    # Initial gradient
    root_grad = F.ones(t.shape, requires_grad=False, backend=t._backend).data
    t._node.grad_buffer = root_grad
    _backward(t._node)
    # return a Tensor wrapping the original root grad
    return Tensor(root_grad, backend=t._backend, requires_grad=False)
