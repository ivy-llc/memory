"""
Collection of tests for templated neural network memory modules
"""

# global
import numpy as np

# local
import ivy_memory as ivy_mem
import ivy_memory_tests.helpers as helpers


def test_lstm():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxsymbolic split returns either list or tensor depending on number of splits
            continue

        # inputs
        b = 2
        t = 3
        input_channels = 4
        hidden_channels = 5
        x = lib.cast(lib.linspace(lib.zeros([b, t]), lib.ones([b, t]), input_channels), 'float32')
        init_h = lib.ones([b, hidden_channels])
        init_c = lib.ones([b, hidden_channels])
        kernel = lib.variable(lib.ones([input_channels, 4*hidden_channels]))*0.5
        recurrent_kernel = lib.variable(lib.ones([hidden_channels, 4*hidden_channels]))*0.5

        # targets
        last_output_true = np.ones([b, hidden_channels]) * 0.9676078
        output_true = np.array([[[0.97068775, 0.97068775, 0.97068775, 0.97068775, 0.97068775],
                                 [0.9653918, 0.9653918, 0.9653918, 0.9653918, 0.9653918],
                                 [0.9676078, 0.9676078, 0.9676078, 0.9676078, 0.9676078]],
                                [[0.97068775, 0.97068775, 0.97068775, 0.97068775, 0.97068775],
                                 [0.9653918, 0.9653918, 0.9653918, 0.9653918, 0.9653918],
                                 [0.9676078, 0.9676078, 0.9676078, 0.9676078, 0.9676078]]], dtype='float32')
        state_h_true = np.ones([b, hidden_channels]) * 0.96644664
        state_c_true = np.ones([b, hidden_channels]) * 3.708991

        last_output, output, state_h, state_c = call(
            ivy_mem.lstm, x, init_h, init_c, kernel, recurrent_kernel, f=lib)

        assert np.allclose(last_output, last_output_true, atol=1e-6)
        assert np.allclose(output, output_true, atol=1e-6)
        assert np.allclose(state_h, state_h_true, atol=1e-6)
        assert np.allclose(state_c, state_c_true, atol=1e-6)
