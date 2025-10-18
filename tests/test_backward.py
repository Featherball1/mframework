import pytest

from mframework.autograd_utils import gradcheck
from mframework.tensor import Tensor
from mframework.backend import Backend


from _internal.opinfo import OpInfo

@pytest.mark.autograd
def test_op_gradcheck(backend: Backend, opinfo: OpInfo):
    if not opinfo.supports_autograd:
        pytest.skip("Op does not support gradient")

    fn = opinfo.op
    xs = opinfo.sample_inputs(backend)[0]

    for inputs in opinfo.sample_inputs(backend):
        gradcheck(fn, inputs, backend=backend)
