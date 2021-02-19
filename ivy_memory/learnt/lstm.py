"""
Ivy LSTM implementation
"""

# global
import mxnet as mx
from ivy.framework_handler import get_framework as _get_framework


def lstm(x, init_h, init_c, kernel, recurrent_kernel, f=None):
    """
    Perform long-short term memory update by unrolling time dimension of input array.

    :param x: input tensor of LSTM layer.
    :type x: array
    :param init_h: initial state tensor for the cell output.
    :type init_h: array
    :param init_c: initial state tensor for the cell hidden state.
    :type init_c: array
    :param kernel: weights for cell kernel.
    :type kernel: array
    :param recurrent_kernel: weights for cell recurrent kernel.
    :type recurrent_kernel: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return:     last_output [batch,units], output tensor for all timesteps [batch,time,units], state_0, state_1
    """

    f = _get_framework(x, f=f)

    # get shapes
    x_shape = x.shape
    batch_size = x_shape[0]
    timesteps = x_shape[1]
    input_channels = x_shape[2]
    x_flat = f.reshape(x, (-1, input_channels))

    # input kernel
    Wi = kernel
    Wi_x = f.reshape(f.matmul(x_flat, Wi), (batch_size, timesteps, -1))
    Wii_x, Wif_x, Wig_x, Wio_x = f.split(Wi_x, 4, -1)

    # recurrent kernel
    Wh = recurrent_kernel

    # lstm states
    ht = init_h
    ct = init_c

    # lstm outputs
    ot = x
    ots_list = list()

    # unrolled time dimension with lstm steps
    for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(f.unstack(Wii_x, axis=1), f.unstack(Wif_x, axis=1),
                                              f.unstack(Wig_x, axis=1), f.unstack(Wio_x, axis=1)):
        htm1 = ht
        ctm1 = ct

        Wh_htm1 = f.matmul(htm1, Wh)
        Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = f.split(Wh_htm1, num_sections=4, axis=-1)

        it = f.sigmoid(Wii_xt + Whi_htm1)
        ft = f.sigmoid(Wif_xt + Whf_htm1)
        gt = f.tanh(Wig_xt + Whg_htm1)
        ot = f.sigmoid(Wio_xt + Who_htm1)
        ct = ft * ctm1 + it * gt
        ht = ot * f.tanh(ct)

        ots_list.append(f.expand_dims(ot, 1))

    return ot, f.concatenate(ots_list, 1), ht, ct
