"""
Implementation of Linear Neural Network Layer
"""

# global
import ivy
from ivy.core.container import Container


class Linear:

    def __init__(self, input_channels, output_channels, v=None):
        self._input_channels = input_channels
        self._output_channels = output_channels
        if v is None:
            wlim = (6 / (output_channels + input_channels)) ** 0.5
            w = ivy.variable(ivy.random_uniform(-wlim, wlim, (output_channels, input_channels)))
            b = ivy.variable(ivy.zeros([output_channels]))
            self.v = Container({'w': w, 'b': b})
        else:
            self.v = Container(v)

    def forward(self, inputs, v=None):
        if v is None:
            v = self.v
        else:
            v = Container(v)
        return ivy.linear(inputs, v.w, v.b)
