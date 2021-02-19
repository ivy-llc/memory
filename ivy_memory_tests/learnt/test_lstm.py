"""
Collection of tests for lstm memory module
"""

# global
import ivy
import numpy as np
from ivy.core.container import Container

# local
import ivy_memory as ivy_mem


def test_lstm_update(dev_str, call):

    # inputs
    b = 2
    t = 3
    input_channels = 4
    hidden_channels = 5
    x = ivy.cast(ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), 'float32')
    init_h = ivy.ones([b, hidden_channels])
    init_c = ivy.ones([b, hidden_channels])
    kernel = ivy.variable(ivy.ones([input_channels, 4*hidden_channels]))*0.5
    recurrent_kernel = ivy.variable(ivy.ones([hidden_channels, 4*hidden_channels]))*0.5

    # targets
    last_output_true = np.ones([b, hidden_channels]) * 0.96644664
    output_true = np.array([[[0.93137765, 0.93137765, 0.93137765, 0.93137765, 0.93137765],
                             [0.9587628, 0.9587628, 0.9587628, 0.9587628, 0.9587628],
                             [0.96644664, 0.96644664, 0.96644664, 0.96644664, 0.96644664]],
                            [[0.93137765, 0.93137765, 0.93137765, 0.93137765, 0.93137765],
                             [0.9587628, 0.9587628, 0.9587628, 0.9587628, 0.9587628],
                             [0.96644664, 0.96644664, 0.96644664, 0.96644664, 0.96644664]]], dtype='float32')
    state_c_true = np.ones([b, hidden_channels]) * 3.708991

    output, state_c = call(
        ivy_mem.lstm_update, x, init_h, init_c, kernel, recurrent_kernel)

    assert np.allclose(output[..., -1, :], last_output_true, atol=1e-6)
    assert np.allclose(output, output_true, atol=1e-6)
    assert np.allclose(state_c, state_c_true, atol=1e-6)


def test_lstm_layer(dev_str, call):

    # inputs
    b = 2
    t = 3
    input_channels = 4
    hidden_channels = 5
    x = ivy.cast(ivy.linspace(ivy.zeros([b, t]), ivy.ones([b, t]), input_channels), 'float32')
    init_h = ivy.ones([b, hidden_channels])
    init_c = ivy.ones([b, hidden_channels])
    kernel = ivy.variable(ivy.ones([input_channels, 4*hidden_channels]))*0.5
    recurrent_kernel = ivy.variable(ivy.ones([hidden_channels, 4*hidden_channels]))*0.5
    v = Container({'input': {'layer_0': {'w': kernel}},
                   'recurrent': {'layer_0': {'w': recurrent_kernel}}})

    # targets
    last_output_true = np.ones([b, hidden_channels]) * 0.96644664
    output_true = np.array([[[0.93137765, 0.93137765, 0.93137765, 0.93137765, 0.93137765],
                             [0.9587628, 0.9587628, 0.9587628, 0.9587628, 0.9587628],
                             [0.96644664, 0.96644664, 0.96644664, 0.96644664, 0.96644664]],
                            [[0.93137765, 0.93137765, 0.93137765, 0.93137765, 0.93137765],
                             [0.9587628, 0.9587628, 0.9587628, 0.9587628, 0.9587628],
                             [0.96644664, 0.96644664, 0.96644664, 0.96644664, 0.96644664]]], dtype='float32')
    state_c_true = np.ones([b, hidden_channels]) * 3.708991

    lstm_layer = ivy_mem.LSTM(input_channels, hidden_channels, v=v)

    output, (state_h, state_c) = call(
        lstm_layer.forward, x, ([init_h], [init_c]))

    assert np.allclose(output[..., -1, :], last_output_true, atol=1e-6)
    assert np.allclose(output, output_true, atol=1e-6)
    assert np.allclose(state_c, state_c_true, atol=1e-6)
