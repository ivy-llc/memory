"""
Collection of tests for linear network layer
"""

# global
import ivy
import pytest
import numpy as np
from ivy_tests import helpers
from ivy.core.container import Container

# local
import ivy_memory as ivy_mem


class LinearTestData:

    def __init__(self):
        self.linear_return = np.array([[0.30230279, 0.65123089, 0.30132881, -0.90954636, 1.08810135]])


td = LinearTestData()


@pytest.mark.parametrize(
    "batch_shape", [[1], [1, 2]])
def test_linear_layer(batch_shape, dev_str, call):

    # inputs
    input_channels = 4
    output_channels = 5
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels), 'float32')

    # linear object wo vars
    linear_layer = ivy_mem.Linear(input_channels, output_channels)

    # test
    assert call(linear_layer.forward, x).shape == tuple(batch_shape + [output_channels])

    # vars
    np.random.seed(0)
    wlim = (6 / (output_channels + input_channels)) ** 0.5
    w = ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, (output_channels, input_channels)), 'float32'))
    b = ivy.variable(ivy.zeros([output_channels]))
    v = Container({'w': w, 'b': b})

    # linear object w vars
    linear_layer = ivy_mem.Linear(input_channels, output_channels, v)

    # test
    assert np.allclose(call(linear_layer.forward, x), td.linear_return)

    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support try-catch statements
        return
    helpers.assert_compilable(linear_layer.forward)
