import pytest
import threading
import time

from mframework.autograd.backend import BackendType, NumpyBackend, CupyBackend
from mframework.state import (
    get_backend,
    set_state,
    reset_state_to_defaults,
    state,
)

# -------------------------------------------------------------
# 1. Default behavior
# -------------------------------------------------------------
def test_default_backend_is_numpy():
    backend = get_backend()
    assert isinstance(backend, NumpyBackend)
    assert backend.backend_type == BackendType.NUMPY


# -------------------------------------------------------------
# 2. Setting backend state
# -------------------------------------------------------------
@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_set_state_changes_backend():
    set_state(BackendType.CUPY)
    backend = get_backend()
    assert isinstance(backend, CupyBackend)
    assert backend.backend_type == BackendType.CUPY

    # Reset to clean state
    reset_state_to_defaults()
    assert isinstance(get_backend(), NumpyBackend)


# -------------------------------------------------------------
# 3. Context manager restores correctly
# -------------------------------------------------------------
@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_state_context_manager_restores_backend():
    # Start from numpy
    assert isinstance(get_backend(), NumpyBackend)
    with state(BackendType.CUPY):
        assert isinstance(get_backend(), CupyBackend)
    # After context exit, should be back to numpy
    assert isinstance(get_backend(), NumpyBackend)


@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_state_context_restores_after_exception():
    assert isinstance(get_backend(), NumpyBackend)
    try:
        with state(BackendType.CUPY):
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    # Still should be numpy
    assert isinstance(get_backend(), NumpyBackend)


# -------------------------------------------------------------
# 4. Thread isolation
# -------------------------------------------------------------
@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_thread_isolation():
    results = {}

    def worker(name, backend_type):
        with state(backend_type):
            results[name] = get_backend().backend_type
            time.sleep(0.1)
        results[f"{name}_after"] = get_backend().backend_type

    t1 = threading.Thread(target=worker, args=("t1", BackendType.NUMPY))
    t2 = threading.Thread(target=worker, args=("t2", BackendType.CUPY))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Each thread's in-context backend is correct
    assert results["t1"] == BackendType.NUMPY
    assert results["t2"] == BackendType.CUPY

    # Each thread's backend after context is reset to numpy
    assert results["t1_after"] == BackendType.NUMPY
    assert results["t2_after"] == BackendType.NUMPY


# -------------------------------------------------------------
# 5. Reset state affects only current thread
# -------------------------------------------------------------
@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_reset_state_is_thread_local():
    results = {}

    def worker():
        set_state(BackendType.CUPY)
        reset_state_to_defaults()
        results["thread_backend"] = get_backend().backend_type

    # Main thread sets CUPY
    set_state(BackendType.CUPY)
    # Spawn another thread that resets its own state
    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # The worker reset only itself to numpy
    assert results["thread_backend"] == BackendType.NUMPY
    # Main thread is still cupy
    assert get_backend().backend_type == BackendType.CUPY

    reset_state_to_defaults()


# -------------------------------------------------------------
# 6. Nested context managers
# -------------------------------------------------------------
@pytest.mark.skip(reason="Requires CUPY backend implemented.")
def test_nested_contexts_restore_correctly():
    reset_state_to_defaults()
    assert isinstance(get_backend(), NumpyBackend)

    with state(BackendType.CUPY):
        assert isinstance(get_backend(), CupyBackend)
        with state(BackendType.NUMPY):
            assert isinstance(get_backend(), NumpyBackend)
        # Inner exited, outer CUPY restored
        assert isinstance(get_backend(), CupyBackend)

    # All exited — back to numpy
    assert isinstance(get_backend(), NumpyBackend)
