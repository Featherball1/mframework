

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
In particular, they contain the parent node and the slot of the function that it fits into. 

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

With the table prepared we can execute backward. Create a ReadyQueue, which keeps
track of nodes with in-degree zero. Then repeatedly:
    1) Pop a ready node from the queue (order does not matter - can even be parallelised)
    2) Compute the local gradient at the ready node
    3) Send gradients to parents
    4) Decrement in_degree counters for each parents
    5) Enqueue parents whose dependency counter hits zero
"""

class Node:
    pass

class Edge:
    """
    Represents a particular input of a function / an edge in the graph. 
    """
    pass

class ReadyQueue:
    """
    In pytorch this is multithreaded, and these
    wrapper methods around a heap would have mutex guards. 

    It is a multi-producer, multi-consumer thread safe queue. 

    We need a special ReadyQueue class because we need ReadyQueues to exist per-device. 
    """

    pass

class Engine:
    """
    Engine implements backpropagation from output variables and their gradients
    to "root" variables, those are, ones with requires_grad=True. 
    """
    
    pass

def backward():
    pass