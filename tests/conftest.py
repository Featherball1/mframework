import pytest

from mframework.backend import NumpyBackend

from _internal.opinfo import op_db

# -- Define available backends --
BACKENDS = ["numpy"]

# -- Fixtures for backends and ops --
@pytest.fixture(params=BACKENDS, scope="session")
def backend(request):
    """Return a backend instance (e.g. numpy, cupy)."""
    if request.param == "numpy":
        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {request.param}")

@pytest.fixture(params=op_db, scope="session")
def opinfo(request):
    """Provide metadata about each operation."""
    return request.param
