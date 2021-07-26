"""
Implementation of Neural Turing Machine
"""

# global
import ivy
import math
import collections
from ivy.core.container import Container

# local
from ivy.neural_net_stateful.layers import Linear, LSTM


NTMControllerState = collections.namedtuple('NTMControllerState',
                                            ('controller_state', 'read_vector_list', 'w_list', 'usage_indicator', 'M'))


def _expand(x, dim, N):
    return ivy.concatenate([ivy.expand_dims(x, dim) for _ in range(N)], dim)


class NTMCell(ivy.Module):
    def __init__(self, controller, controller_proj, output_proj, output_dim, ctrl_input_size, ctrl_output_size,
                 total_parameter_num, memory_size, memory_vector_dim, read_head_num, write_head_num, v=None, usage=None,
                 addressing_mode='content_and_location', shift_range=1, clip_value=20, init_value=1e-6,
                 sequential_writing=False, retroactive_updates=False, retroactive_discount=0.96, with_erase=True,
                 seed=0):

        # vanilla ntm
        self._memory_size = memory_size
        self._memory_vector_dim = memory_vector_dim
        self._read_head_num = read_head_num
        self._write_head_num = write_head_num
        self._init_value = init_value
        self._addressing_mode = addressing_mode
        self._clip_value = clip_value
        self._output_dim = output_dim
        self._shift_range = shift_range
        self._num_parameters_per_head = self._memory_vector_dim + 1 + 1 + (self._shift_range * 2 + 1) + 1
        self._num_heads = self._read_head_num + self._write_head_num
        ivy.seed(seed)

        # fns + classes
        self._controller = controller
        self._controller_proj = controller_proj
        self._output_proj = output_proj

        # usage
        if usage is not None:
            self._usage = usage
        else:
            self._usage = ivy.zeros([memory_size, ])

        # step
        self._step = 0

        # MERLIN changes
        self._sequential_writing = sequential_writing
        self._retroactive_updates = retroactive_updates
        self._retroactive_discount = retroactive_discount
        self._with_erase = with_erase

        # variables
        ivy.Module.__init__(self, 'cpu')

    def _create_variables(self, dev_str):
        vars_dict = dict()
        wlim = (6 / (2 * self._memory_vector_dim)) ** 0.5
        vars_dict['read_weights'] =\
            dict(zip(['w_' + str(i) for i in range(self._read_head_num)],
                     [ivy.variable(ivy.random_uniform(-wlim, wlim, [self._memory_vector_dim, ], dev_str=dev_str))
                      for _ in range(self._read_head_num)]))
        wlim = (6 / (2 * self._memory_size)) ** 0.5
        vars_dict['write_weights'] =\
            dict(zip(['w_' + str(i) for i in range(self._read_head_num + self._write_head_num)],
                     [ivy.variable(ivy.random_uniform(-wlim, wlim, [self._memory_size, ], dev_str=dev_str))
                      for _ in range(self._read_head_num + self._write_head_num)]))
        vars_dict['memory'] = ivy.variable(
            ivy.ones([self._memory_size, self._memory_vector_dim], dev_str=dev_str) * self._init_value)
        return vars_dict

    def _addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = ivy.expand_dims(k, axis=2)
        inner_product = ivy.matmul(prev_M, k)
        k_norm = ivy.reduce_sum(k ** 2, axis=1, keepdims=True) ** 0.5
        M_norm = ivy.reduce_sum(prev_M ** 2, axis=2, keepdims=True) ** 0.5
        norm_product = M_norm * k_norm
        K = ivy.squeeze(inner_product / (norm_product + 1e-8))  # eq (6)

        # Calculating w^c

        K_amplified = ivy.exp(ivy.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / ivy.reduce_sum(K_amplified, axis=1, keepdims=True)  # eq (5)

        if self._addressing_mode == 'content':  # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = ivy.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w  # eq (7)

        s = ivy.concatenate([s[:, :self._shift_range + 1],
                                 ivy.zeros([s.shape[0], self._memory_size - (self._shift_range * 2 + 1)]),
                                 s[:, -self._shift_range:]], axis=1)
        t = ivy.concatenate([ivy.flip(s, axis=[1]),
                                 ivy.flip(s, axis=[1])], axis=1)
        s_matrix = ivy.stack(
            [t[:, self._memory_size - i - 1:self._memory_size * 2 - i - 1] for i in range(self._memory_size)],
            axis=1)
        w_ = ivy.reduce_sum(ivy.expand_dims(w_g, axis=1) * s_matrix, axis=2)  # eq (8)
        w_sharpen = w_ ** ivy.expand_dims(gamma, axis=1)
        w = w_sharpen / ivy.reduce_sum(w_sharpen, axis=1, keepdims=True)  # eq (9)

        return w

    # Public Methods #
    # ---------------#

    def get_start_state(self, _=None, batch_size=None, dtype_str=None, v=None):
        if v is None:
            v = self.v
        else:
            v = Container(v)
        read_vector_list = [_expand(ivy.tanh(var), dim=0, N=batch_size)
                            for _, var in v.read_weights.to_iterator()]
        w_list = [_expand(ivy.softmax(var), dim=0, N=batch_size)
                  for _, var in v.write_weights.to_iterator()]
        usage_indicator = _expand(self._usage, dim=0, N=batch_size)
        M = _expand(v.memory, dim=0, N=batch_size)
        return NTMControllerState(
            controller_state=self._controller.get_initial_state(batch_shape=(batch_size,)),
            read_vector_list=read_vector_list,
            w_list=w_list,
            usage_indicator=usage_indicator,
            M=M)

    def _forward(self, x, prev_state):
        prev_read_vector_list = prev_state[1]

        controller_input = ivy.concatenate([x] + prev_read_vector_list, axis=1)
        controller_output, controller_state = self._controller(ivy.expand_dims(controller_input, -2),
                                                               initial_state=prev_state[0])
        controller_output = controller_output[..., -1, :]

        parameters = self._controller_proj(controller_output)
        parameters = ivy.clip(parameters, -self._clip_value, self._clip_value)
        head_parameter_list = \
            ivy.split(parameters[:, :self._num_parameters_per_head * self._num_heads], self._num_heads,
                          axis=1)
        erase_add_list = ivy.split(parameters[:, self._num_parameters_per_head * self._num_heads:],
                                       2 * self._write_head_num, axis=1)

        prev_w_list = prev_state[2]
        prev_M = prev_state[4]
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = ivy.tanh(head_parameter[:, 0:self._memory_vector_dim])
            beta = ivy.softplus(head_parameter[:, self._memory_vector_dim])
            g = ivy.sigmoid(head_parameter[:, self._memory_vector_dim + 1])
            s = ivy.softmax(
                head_parameter[:, self._memory_vector_dim + 2:self._memory_vector_dim +
                                                              2 + (self._shift_range * 2 + 1)])
            gamma = ivy.softplus(head_parameter[:, -1]) + 1
            w = self._addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        # Reading (Sec 3.1)

        read_w_list = w_list[:self._read_head_num]
        if self._step == 0:
            usage_indicator = ivy.zeros_like(w_list[0])
        else:
            usage_indicator = prev_state[3] + ivy.reduce_sum(ivy.concatenate(read_w_list, 0))
        read_vector_list = []
        for i in range(self._read_head_num):
            read_vector = ivy.reduce_sum(ivy.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        prev_wrtie_w_list = prev_w_list[self._read_head_num:]
        w_wr_size = math.ceil(self._memory_size / 2) if self._retroactive_updates else self._memory_size
        if self._sequential_writing:
            batch_size = ivy.shape(x)[0]
            if self._step < w_wr_size:
                w_wr_list = [ivy.tile(ivy.cast(ivy.one_hot(
                    ivy.array([self._step]), w_wr_size), 'float32'),
                    (batch_size, 1))] * self._write_head_num
            else:
                batch_idxs = ivy.expand_dims(ivy.arange(batch_size, 0), -1)
                mem_idxs = ivy.expand_dims(ivy.argmax(usage_indicator[..., :w_wr_size], -1), -1)
                total_idxs = ivy.concatenate((batch_idxs, mem_idxs), -1)
                w_wr_list = [ivy.scatter_nd(total_idxs, ivy.ones((batch_size,)),
                                                (batch_size, w_wr_size))] * self._write_head_num
        else:
            w_wr_list = w_list[self._read_head_num:]
        if self._retroactive_updates:
            w_ret_list = [self._retroactive_discount * prev_wrtie_w[..., w_wr_size:] +
                          (1 - self._retroactive_discount) * prev_wrtie_w[..., :w_wr_size]
                          for prev_wrtie_w in prev_wrtie_w_list]
            w_wrtie_list = [ivy.concatenate((w_wr, w_ret), -1) for w_wr, w_ret in zip(w_wr_list, w_ret_list)]
        else:
            w_wrtie_list = w_wr_list
        M = prev_M
        for i in range(self._write_head_num):
            w = ivy.expand_dims(w_wrtie_list[i], axis=2)
            if self._with_erase:
                erase_vector = ivy.expand_dims(ivy.sigmoid(erase_add_list[i * 2]), axis=1)
                M = M * ivy.ones(ivy.shape(M)) - ivy.matmul(w, erase_vector)
            add_vector = ivy.expand_dims(ivy.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M + ivy.matmul(w, add_vector)

        NTM_output = self._output_proj(ivy.concatenate([controller_output] + read_vector_list, axis=1))
        NTM_output = ivy.clip(NTM_output, -self._clip_value, self._clip_value)

        self._step += 1
        return NTM_output, NTMControllerState(
            controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list,
            usage_indicator=usage_indicator, M=M)

    # Properties #
    # -----------#

    @property
    def _state_size(self):
        return NTMControllerState(
            controller_state=self._controller.state_size[0],
            read_vector_list=[self._memory_vector_dim for _ in range(self._read_head_num)],
            w_list=[self._memory_size for _ in range(self._read_head_num + self._write_head_num)],
            usage_indicator=[self._memory_size],
            M=[self._memory_size, self._memory_vector_dim])

    @property
    def _output_size(self):
        return self._output_dim


class NTM(ivy.Module):

    def __init__(self, input_dim, output_dim, ctrl_output_size, ctrl_layers, memory_size, memory_vector_dim,
                 read_head_num, write_head_num, v=None, usage=None, addressing_mode='content_and_location',
                 shift_range=1, clip_value=20, init_value=1e-6, sequential_writing=False,
                 retroactive_updates=False, retroactive_discount=0.96, with_erase=True):
        ctrl_input_size = read_head_num * memory_vector_dim + input_dim
        num_heads = read_head_num + write_head_num
        num_parameters_per_head = memory_vector_dim + 1 + 1 + (shift_range * 2 + 1) + 1
        total_parameter_num = num_parameters_per_head * num_heads + memory_vector_dim * 2 * write_head_num
        ctrl_v = v.ntm_cell.controller if \
            v is not None and 'ntm_cell' in v and 'controller' in v.ntm_cell else None
        ctrl = LSTM(ctrl_input_size, ctrl_output_size, num_layers=ctrl_layers, v=ctrl_v)
        ctrl_proj_v = v.ntm_cell.controller_proj \
            if v is not None and 'ntm_cell' in v and 'controller_proj' in v.ntm_cell else None
        ctrl_proj = Linear(ctrl_output_size, total_parameter_num, v=ctrl_proj_v)
        out_proj_v = v.ntm_cell.output_proj \
            if v is not None and 'ntm_cell' in v and 'output_proj' in v.ntm_cell else None
        out_proj = Linear(ctrl_output_size + read_head_num * memory_vector_dim, output_dim, v=out_proj_v)

        ntm_v = v.ntm if v is not None and 'ntm' in v else None
        ntm_cell = NTMCell(ctrl, ctrl_proj, out_proj, output_dim, ctrl_input_size, ctrl_output_size,
                           total_parameter_num, memory_size, memory_vector_dim, read_head_num, write_head_num, ntm_v,
                           usage, addressing_mode, shift_range, clip_value, init_value, sequential_writing,
                           retroactive_updates, retroactive_discount, with_erase)
        self._ntm_cell = ntm_cell
        ivy.Module.__init__(self, 'cpu', v=v)

    def _forward(self, inputs, hidden=None):
        inputs_shape = list(inputs.shape)
        batch_shape = inputs_shape[:-2]
        time_dim = inputs_shape[-2]
        inputs = ivy.reshape(inputs, [-1] + list(inputs.shape[-2:]))
        if hidden is None:
            hidden = self._ntm_cell.get_start_state(inputs, inputs.shape[0], inputs.dtype, v=self.v.ntm_cell)
        outputs = []
        for x in ivy.unstack(inputs, 1):
            output, hidden = self._ntm_cell(x, hidden)
            outputs.append(output)
        ret = ivy.stack(outputs, 1)
        return ivy.reshape(ret, batch_shape + [time_dim, -1])
