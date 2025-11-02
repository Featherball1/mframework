"""
To enhance the experience of using the framework it maintains a global state.
This allows us to omit `backend=` arguments and fall back to a default backend, which is a module-level global.

It can be overriden at any time with a `with` keyword, i.e.

with state(BackendType.CUPY):
    ... do things ...

and it can be reset at any time using reset_state()

The default backend is numpy. 

The state is always made thread-local. 
"""

from contextlib import contextmanager
import threading

from mframework.autograd.backend import (
    Backend,
    CupyBackend,
    NumpyBackend,
    BackendType
)

# Thread-local storage container
_thread_local = threading.local()

def get_backend() -> Backend:
    """Return the current thread's backend, defaulting to NumpyBackend."""
    backend = getattr(_thread_local, "backend", None)
    if backend is None:
        backend = NumpyBackend()
        _thread_local.backend = backend
    return backend


def _set_backend(backend: Backend):
    """Set the backend for the current thread."""
    _thread_local.backend = backend


def set_state(backend_type: BackendType):
    """Set backend by type for the current thread."""
    if backend_type == BackendType.NUMPY:
        _set_backend(NumpyBackend())
    elif backend_type == BackendType.CUPY:
        _set_backend(CupyBackend())
    else:
        raise ValueError(f"Unrecognized backend type: {backend_type}")


def reset_state_to_defaults():
    """Reset this thread's backend to the default (numpy)."""
    _set_backend(NumpyBackend())


@contextmanager
def state(backend_type: BackendType):
    """Context manager to temporarily change backend within this thread."""
    prev_backend = get_backend()
    set_state(backend_type)
    try:
        yield
    finally:
        _set_backend(prev_backend)
