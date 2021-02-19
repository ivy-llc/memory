"""
Ivy LSTM implementation
"""

# global
import ivy
from ivy.core.container import Container

# local
import ivy_memory as ivy_mem


# Functional API
# -------------#

def lstm_update(x, init_h, init_c, kernel, recurrent_kernel, bias=None, recurrent_bias=None):
    """
    Perform long-short term memory update by unrolling time dimension of input array.

    :param x: input tensor of LSTM layer *[batch_shape, t, in]*.
    :type x: array
    :param init_h: initial state tensor for the cell output *[batch_shape, out]*.
    :type init_h: array
    :param init_c: initial state tensor for the cell hidden state *[batch_shape, out]*.
    :type init_c: array
    :param kernel: weights for cell kernel *[in, 4 x out]*.
    :type kernel: array
    :param recurrent_kernel: weights for cell recurrent kernel *[out, 4 x out]*.
    :type recurrent_kernel: array
    :param bias: bias for cell kernel *[4 x out]*.
    :type bias: array
    :param recurrent_bias: bias for cell recurrent kernel *[4 x out]*.
    :type recurrent_bias: array
    :return: hidden state for all timesteps *[batch_shape,t,out]* and cell state for last timestep *[batch_shape,out]*
    """

    # get shapes
    x_shape = list(x.shape)
    batch_shape = x_shape[:-2]
    timesteps = x_shape[-2]
    input_channels = x_shape[-1]
    x_flat = ivy.reshape(x, (-1, input_channels))

    # input kernel
    Wi = kernel
    Wi_x = ivy.reshape(ivy.matmul(x_flat, Wi) + (bias if bias is not None else 0),
                        batch_shape + [timesteps, -1])
    Wii_x, Wif_x, Wig_x, Wio_x = ivy.split(Wi_x, 4, -1)

    # recurrent kernel
    Wh = recurrent_kernel

    # lstm states
    ht = init_h
    ct = init_c

    # lstm outputs
    ot = x
    hts_list = list()

    # unrolled time dimension with lstm steps
    for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(ivy.unstack(Wii_x, axis=-2), ivy.unstack(Wif_x, axis=-2),
                                              ivy.unstack(Wig_x, axis=-2), ivy.unstack(Wio_x, axis=-2)):
        htm1 = ht
        ctm1 = ct

        Wh_htm1 = ivy.matmul(htm1, Wh) + (recurrent_bias if recurrent_bias is not None else 0)
        Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = ivy.split(Wh_htm1, num_sections=4, axis=-1)

        it = ivy.sigmoid(Wii_xt + Whi_htm1)
        ft = ivy.sigmoid(Wif_xt + Whf_htm1)
        gt = ivy.tanh(Wig_xt + Whg_htm1)
        ot = ivy.sigmoid(Wio_xt + Who_htm1)
        ct = ft * ctm1 + it * gt
        ht = ot * ivy.tanh(ct)

        hts_list.append(ivy.expand_dims(ht, -2))

    return ivy.concatenate(hts_list, -2), ct


# Classes #
# --------#

class LSTM:

    def __init__(self, input_channels, output_channels, num_layers=1, return_sequence=True, return_state=True, v=None):
        """
        Construct LSTM layer

        :param input_channels: Number of input channels for the layer
        :type input_channels: int
        :param output_channels: Number of output channels for the layer
        :type output_channels: int
        :param num_layers: Number of lstm cells in the lstm layer, default is 1.
        :type num_layers: int, optional
        :param return_sequence: Whether or not to return the entire output sequence, or just the latest timestep.
                                Default is True.
        :type return_sequence: bool, optional
        :param return_state: Whether or not to return the latest hidden and cell states. Default is True.
        :type return_state: bool, optional
        :param v: the variables for each of the lstm cells, as a container, constructed internally by default.
        :type v: ivy container of parameter arrays, optional
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._num_layers = num_layers
        self._return_sequence = return_sequence
        self._return_state = return_state
        if v is None:
            wlim = (6 / (output_channels + input_channels)) ** 0.5
            input_weights = dict(zip(
                ['layer_' + str(i) for i in range(num_layers)],
                [{'w': ivy.variable(ivy.random_uniform(
                    -wlim, wlim, (input_channels if i == 0 else output_channels, 4 * output_channels)))}
                 for i in range(num_layers)]))
            wlim = (6 / (output_channels + output_channels)) ** 0.5
            recurrent_weights = dict(zip(
                ['layer_' + str(i) for i in range(num_layers)],
                [{'w': ivy.variable(ivy.random_uniform(-wlim, wlim, (output_channels, 4 * output_channels)))}
                 for i in range(num_layers)]))
            self.v = Container({'input': input_weights, 'recurrent': recurrent_weights})
        else:
            self.v = Container(v)

    def forward(self, inputs, initial_state=None, v=None):
        """
        Perform forward pass of the lstm layer, which is a set of stacked lstm cells.

        :param inputs: Inputs to process *[batch_shape, t, in]*.
        :type inputs: array
        :param initial_state: 2-tuple of lists of the hidden states h and c for each layer, each of dimension *[batch_shape,out]*.
                        Created internally if None.
        :type initial_state: tuple of list of arrays, optional
        :param v: the variables for each of the lstm cells, as a container, use internal variables by default.
        :type v: ivy container of parameter arrays, optional
        :return: The outputs of the final lstm layer *[batch_shape, t, out]* and the hidden state tuple of lists,
                each of dimension *[batch_shape, out]*
        """
        if v is None:
            v = self.v
        else:
            v = Container(v)
        if initial_state is None:
            initial_state = self.get_initial_state(inputs.shape[:-2])
        h_n_list = list()
        c_n_list = list()
        h_t = inputs
        for h_0, c_0, (_, lstm_input_var), (_, lstm_recurrent_var) in zip(
                initial_state[0], initial_state[1], v.input.items(), v.recurrent.items()):
            h_t, c_n = ivy_mem.lstm_update(h_t, h_0, c_0, lstm_input_var.w, lstm_recurrent_var.w)
            h_n_list.append(h_t[..., -1, :])
            c_n_list.append(c_n)
        if not self._return_sequence:
            h_t = h_t[..., -1, :]
        if not self._return_state:
            return h_t
        return h_t, (h_n_list, c_n_list)

    def get_initial_state(self, batch_shape):
        batch_shape = list(batch_shape)
        return ([ivy.zeros((batch_shape + [self._output_channels])) for i in range(self._num_layers)],
                [ivy.zeros((batch_shape + [self._output_channels])) for i in range(self._num_layers)])
