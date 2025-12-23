from typing import Any, Callable, Self, Iterator
from abc import ABC, abstractmethod

from mframework.autograd.tensor import Tensor, Parameter, Buffer

"""
Modules.

A Module is a container for 
     - Parameters
     - Buffers
     - Other Modules.

A Parameter in the model is a tensor that can be tuned during training.
A Buffer is a tensor that is used to help maintain Module state, and not tuned during training.
Other modules are submodules of the current module.

The idea behind a module is to abstract away many individual tensors/submodules into a bundle called a module
that allows for easy end-user manipulation.
"""

class Module(ABC):
    def __init__(self) -> None:
        self._training = True
        self._submodules: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}
        self._buffers: dict[str, Tensor] = {}

    def register_parameter(self, name: str, param: Parameter) -> None:
        """
        Register a named parameter to the module.
        TODO: pytorch would do more type checking / general error handling here. 
        """
        self._parameters[name] = param
    
    def register_buffer(self, name: str, buffer: Buffer) -> None:
        """
        Register a named parameter to the module.
        TODO: pytorch would do more type checking / general error handling here. 
        """
        self._buffers[name] = buffer

    def register_submodule(self, name: str, module: "Module") -> None:
        """
        Register a named parameter to the module.
        TODO: pytorch would do more type checking / general error handling here. 
        """
        self._submodules[name] = module

    def parameters(self) -> Iterator[Parameter]:
        """
        Retrieve an iterator over all parameters in the module as well as its submodules.
        Traverses submodules recursively.
        """
        for p in self._parameters.values():
            yield p
        for m in self._submodules.values():
            yield from m.parameters()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom setattr is for ergonomics for automatically register parameters, buffers and submodules.
        Were it not here we would need to provide a method such as
        module.register_parameter("weight", Parameter(...))
        which is not ergonomic. 
        """
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_submodule(name, value)
        elif isinstance(value, Buffer):
            self.register_buffer(name, value)
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            if name in self._parameters:
                return self._parameters[name]
        if '_buffers' in self.__dict__:
            if name in self._buffers:
                return self._buffers[name]
        if '_submodules' in self.__dict__:
            if name in self._submodules:
                return self._submodules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    """
    Module apply behaviour. 

    TODO: implement these methods when needed later on. See Pytorch source code for motivation. 
    """

    def apply(self, f: Callable[["Module"], None]) -> Self:
        """
        Apply a function recursively to every submodule as well as self.
        Typical use case would be 
            - to initialize parameters of a model.
            - to move parameters to a different backend.
            - to change types of all parameters / buffers.
        """
        f(self)
        for m in self._submodules.values():
            m.apply(f)
        return self
    
    """
    Train / eval mode switching. 

    We have a self._train: bool flag and recursively implement:
        train()
        eval()
    methods that switch this flag. The purpose of this is for layers like Dropout
    that behave differently during training and evaluation.
    """

    def train(self, mode: bool = True):
        self._training = mode
        for m in self._submodules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    """
    Forward functionality must be implemented by subclasses.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
