from mframework.tensor import Tensor, Parameter

"""
Modules.

A Module is a container for Parameters and other Modules.

The idea behind a module is to abstract away many individual tensors/submodules into a bundle called a module
that allows for easy end-user manipulation.

Module
---------------------
| Parameters        |
| Tensors           |
| Submodules        |
|                   |
| Forward           |
---------------------

"""

class Module:
    def __init__(self) -> None:
        submodules = []
        parameters = []

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
