"""
Collection of tests for ntm memory module
"""

# global
import ivy
import pytest
import numpy as np
from ivy_tests import helpers
from ivy.core.container import Container

# local
import ivy_memory as ivy_mem


class NTMTestData:

    def __init__(self):
        self.ntm_return = dict()
        self.ntm_return['content'] = \
            np.array(
            [[[0.00869071, 0.01375137, -0.03888849, -0.0062099,
               0.04832713, 0.02295336, -0.03013051, 0.00841217],
              [0.03594615, 0.03818635, -0.05962976, -0.01571055,
               0.09582922, 0.04535963, -0.05843233, 0.0218056],
              [0.06523251, 0.0621037, -0.06750974, -0.02423633,
               0.13218611, 0.06428179, -0.07842496, 0.03328294],
              [0.08936682, 0.08157387, -0.06947725, -0.0309606,
               0.15841076, 0.07914014, -0.09124488, 0.04102223],
              [0.107168, 0.09602168, -0.06931666, -0.03598689,
               0.17695096, 0.09056101, -0.09919222, 0.04551058]]])
        self.ntm_return['content_and_location'] = \
            np.array(
            [[[0.00869071, 0.01375137, -0.03888849, -0.0062099,
                0.04832713, 0.02295336, -0.03013051, 0.00841217],
               [0.03634587, 0.03792483, -0.05942222, -0.0152731,
                0.09526038, 0.04544635, -0.05877383, 0.02198462],
               [0.06544678, 0.06188314, -0.06736284, -0.02396519,
                0.13175775, 0.06439733, -0.07864732, 0.03340202],
               [0.08930202, 0.08119108, -0.0692278, -0.03057213,
                0.15766081, 0.07966205, -0.09161495, 0.04123296],
               [0.10764606, 0.09447527, -0.06900684, -0.03614915,
                0.17603959, 0.08985719, -0.09915707, 0.04555906]]])


td = NTMTestData()


@pytest.mark.parametrize(
    "addressing_mode", ['content', 'content_and_location'])
@pytest.mark.parametrize(
    "batch_shape", [[1], [1, 2]])
def test_ntm(addressing_mode, batch_shape, dev_str, call):

    # ntm config
    input_dim = 256
    output_dim = 8
    ctrl_output_size = 256
    ctrl_layers = 2
    memory_size = 5
    timesteps = 5
    memory_vector_dim = 2
    read_head_num = 3
    write_head_num = 1
    shift_range = 0
    clip_value = 20
    init_value = 1e-6
    ctrl_input_size = read_head_num * memory_vector_dim + input_dim
    num_heads = read_head_num + write_head_num
    num_parameters_per_head = memory_vector_dim + 1 + 1 + (shift_range * 2 + 1) + 1
    total_parameter_num = num_parameters_per_head * num_heads + memory_vector_dim * 2 * write_head_num
    usage = ivy.zeros([memory_size, ])

    # memory object wo vars
    ntm = ivy_mem.NTM(
        input_dim, output_dim, ctrl_output_size, ctrl_layers, memory_size, memory_vector_dim,
        read_head_num, write_head_num, addressing_mode=addressing_mode, shift_range=shift_range,
        clip_value=clip_value, sequential_writing=True, retroactive_updates=False, with_erase=False)

    # test
    x = ivy.ones(batch_shape + [timesteps, input_dim])
    assert call(ntm, x).shape == tuple(batch_shape + [timesteps, output_dim])

    # variables
    variables = dict()
    variables['ntm_cell'] = dict()
    np.random.seed(0)

    # lstm
    in_wlim = (6 / (ctrl_input_size + 4 * ctrl_output_size)) ** 0.5
    rec_wlim = (6 / (ctrl_output_size + 4 * ctrl_output_size)) ** 0.5
    variables['ntm_cell']['controller'] = \
        {'input': {'layer1': {'w': ivy.array(np.random.uniform(
            -in_wlim, in_wlim, size=[ctrl_input_size, 4 * ctrl_output_size]).astype(np.float32))},
                   'layer2': {'w': ivy.array(np.random.uniform(
                       -in_wlim, in_wlim, size=[ctrl_output_size, 4 * ctrl_output_size]).astype(np.float32))}},
         'recurrent': {'layer1': {'w': ivy.array(np.random.uniform(
             -rec_wlim, rec_wlim, size=[ctrl_output_size, 4 * ctrl_output_size]).astype(np.float32))},
                       'layer2': {'w': ivy.array(np.random.uniform(
                           -rec_wlim, rec_wlim, size=[ctrl_output_size, 4 * ctrl_output_size]).astype(
                           np.float32))}}}

    # fully connected
    proj_wlim = (6 / (total_parameter_num + ctrl_output_size)) ** 0.5
    variables['ntm_cell']['controller_proj'] = {'w': ivy.array(np.random.uniform(
        -proj_wlim, proj_wlim, size=[total_parameter_num, ctrl_output_size]).astype(np.float32)),
                              'b': ivy.zeros([total_parameter_num])}

    out_wlim = (6 / (total_parameter_num + ctrl_input_size)) ** 0.5
    variables['ntm_cell']['output_proj'] = {'w': ivy.array(np.random.uniform(
        -out_wlim, out_wlim, size=[output_dim, ctrl_output_size + read_head_num * memory_vector_dim]).astype(
        np.float32)),
                             'b': ivy.zeros([output_dim])}

    # memory
    wlim = (6 / (2 * memory_vector_dim)) ** 0.5
    variables['ntm_cell']['read_weights'] = dict(zip(
        ['w_' + str(i) for i in range(read_head_num)],
        [ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, [memory_vector_dim, ]), 'float32'))
         for _ in range(read_head_num)]))

    wlim = (6 / (2 * memory_size)) ** 0.5
    variables['ntm_cell']['write_weights'] = dict(zip(
        ['w_' + str(i) for i in range(read_head_num + write_head_num)],
        [ivy.variable(ivy.array(np.random.uniform(-wlim, wlim, [memory_size, ]), 'float32'))
         for _ in range(read_head_num + write_head_num)]))

    variables['ntm_cell']['memory'] = ivy.variable(ivy.ones([memory_size, memory_vector_dim]) * init_value)

    # memory object w vars
    ntm = ivy_mem.NTM(
        input_dim, output_dim, ctrl_output_size, ctrl_layers, memory_size, memory_vector_dim,
        read_head_num, write_head_num, Container(variables), usage, addressing_mode=addressing_mode,
        shift_range=shift_range, clip_value=clip_value, init_value=init_value, sequential_writing=True,
        retroactive_updates=False, with_erase=False)

    # test
    assert np.allclose(call(ntm, x), td.ntm_return[addressing_mode], atol=1e-6)

    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not support try-catch statements
        return
    helpers.assert_compilable(ntm)
