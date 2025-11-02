from typing import OrderedDict, Iterator

from mframework.nn.module import Module

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name: str, module: "Module") -> None:
        self._submodules[name] = module

    def __iter__(self) -> Iterator[Module]:
        return iter(self._submodules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input
