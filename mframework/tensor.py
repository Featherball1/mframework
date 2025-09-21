from mframework.backend import Backend, backend_dtype

class Tensor:
    def __init__(
        self,
        data: backend_dtype,
        backend: Backend,
        requires_grad: bool = False
    ) -> None:
        self._data = data
        self._backend = backend
        self._requires_grad = requires_grad

    def __add__(self, other: "Tensor") -> "Tensor":
        # Things that could go wrong
            # "other" is not a tensor
            # "other" does not have the same backend as self
        return Tensor(self._backend.add(self._data, other._data), self._backend)

class Parameter(Tensor):
    pass