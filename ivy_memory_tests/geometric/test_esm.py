"""
Collection of simple tests for ESM module
"""

# global
import ivy
import time
import pytest
import ivy_vision
import numpy as np
import ivy_tests.helpers as helpers
from ivy.core.container import Container

# local
from ivy_memory.geometric.esm.esm import ESM


# Helpers #
# --------#

def _get_dummy_obs(batch_size, num_frames, num_cams, image_dims, num_feature_channels, dev_str='cpu'):

    uniform_pixel_coords =\
        ivy_vision.create_uniform_pixel_coords_image(image_dims, [batch_size, num_frames], dev_str=dev_str)

    img_meas = dict()
    for i in range(num_cams):
        img_meas['dummy_cam_{}'.format(i)] =\
            {'img_mean':
                 ivy.concatenate((uniform_pixel_coords[..., 0:2], ivy.random_uniform(
                     1e-3, 1, [batch_size, num_frames] + image_dims + [1 + num_feature_channels],
                     dev_str)), -1),
             'img_var':
                 ivy.random_uniform(
                     1e-3, 1, [batch_size, num_frames] + image_dims + [1 + num_feature_channels], dev_str),
             'validity_mask': ivy.ones([batch_size, num_frames] + image_dims + [1], dev_str=dev_str),
             'pose_mean': ivy.random_uniform(1e-3, 1, [batch_size, num_frames, 6], dev_str),
             'pose_cov': ivy.random_uniform(1e-3, 1, [batch_size, num_frames, 6, 6], dev_str),
             'cam_rel_mat': ivy.identity(4, batch_shape=[batch_size, num_frames], dev_str=dev_str)[..., 0:3, :]}

    return Container({'img_meas': img_meas,
                      'control_mean': ivy.random_uniform(1e-3, 1, [batch_size, num_frames, 6], dev_str),
                      'control_cov': ivy.random_uniform(1e-3, 1, [batch_size, num_frames, 6, 6], dev_str),
                      'agent_rel_mat': ivy.identity(4, batch_shape=[batch_size, num_frames],
                                                    dev_str=dev_str)[..., 0:3, :]})


# PyTorch #
# --------#

def test_construction(dev_str, call):
    if call in [helpers.mx_call]:
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    ESM()


@pytest.mark.parametrize(
    "with_args", [True, False])
def test_inference(with_args, dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    batch_size = 5
    num_timesteps = 6
    num_cams = 7
    num_feature_channels = 3
    image_dims = [3, 3]
    esm = ESM()
    esm.forward(_get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels),
                esm.empty_memory(batch_size, num_timesteps) if with_args else None,
                batch_size=batch_size if with_args else None,
                num_timesteps=num_timesteps if with_args else None,
                num_cams=num_cams if with_args else None,
                image_dims=image_dims if with_args else None)


def test_realtime_speed(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    if call is not helpers.torch_call:
        # pytorch currently the only framework building correctly with gpu
        pytest.skip()
    ivy.seed(0)
    device = 'cpu'
    batch_size = 1
    num_timesteps = 1
    num_cams = 1
    num_feature_channels = 3
    image_dims = [64, 64]
    omni_img_dims = [90, 180]
    esm = ESM(omni_image_dims=omni_img_dims, device=device)
    memory = esm.empty_memory(batch_size, num_timesteps)
    start_time = time.perf_counter()
    for i in range(150):
        obs = _get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels, device)
        memory = esm.forward(obs, memory, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                             image_dims=image_dims)
        memory_mean = memory.mean.numpy()
        assert memory_mean.shape == tuple([batch_size, num_timesteps] + omni_img_dims + [3 + num_feature_channels])
        assert memory_mean[0, 0, 0, 0, 0] == 0.
        np.max(memory_mean)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    assert time_taken < 5.
