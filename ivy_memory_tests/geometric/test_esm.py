"""
Collection of simple tests for ESM module
"""

# global
import os
import ivy
import time
import pytest
import ivy_mech
import ivy_vision
import numpy as np
from ivy_tests.test_ivy import helpers

# local
from ivy_memory.geometric.esm import ESM


# Helpers #
# --------#

def _get_dummy_obs(batch_size, num_frames, num_cams, image_dims, num_feature_channels, dev_str=None, ones=False,
                   empty=False):

    dev_str = ivy.default_device(dev_str)

    uniform_pixel_coords =\
        ivy_vision.create_uniform_pixel_coords_image(image_dims, [batch_size, num_frames], dev_str=dev_str)

    img_meas = dict()
    for i in range(num_cams):
        validity_mask = ivy.ones([batch_size, num_frames] + image_dims + [1], device=dev_str)
        if ones:
            img_mean = ivy.concat((uniform_pixel_coords[..., 0:2], ivy.ones(
                [batch_size, num_frames] + image_dims + [1 + num_feature_channels], device=dev_str)), axis=-1)
            img_var = ivy.ones(
                     [batch_size, num_frames] + image_dims + [3 + num_feature_channels], device=dev_str)*1e-3
            pose_mean = ivy.zeros([batch_size, num_frames, 6], device=dev_str)
            pose_cov = ivy.ones([batch_size, num_frames, 6, 6], device=dev_str)*1e-3
        else:
            img_mean = ivy.concat((uniform_pixel_coords[..., 0:2], ivy.random_uniform(
                low=1e-3, high=1, shape=[batch_size, num_frames] + image_dims + [1 + num_feature_channels], device=dev_str)), axis=-1)
            img_var = ivy.random_uniform(
                     low=1e-3, high=1, shape=[batch_size, num_frames] + image_dims + [3 + num_feature_channels], device=dev_str)
            pose_mean = ivy.random_uniform(low=1e-3, high=1, shape=[batch_size, num_frames, 6], device=dev_str)
            pose_cov = ivy.random_uniform(low=1e-3, high=1, shape=[batch_size, num_frames, 6, 6], device=dev_str)
        if empty:
            img_var = ivy.ones_like(img_var) * 1e12
            validity_mask = ivy.zeros_like(validity_mask)
        img_meas['dummy_cam_{}'.format(i)] =\
            {'img_mean': img_mean,
             'img_var': img_var,
             'validity_mask': validity_mask,
             'pose_mean': pose_mean,
             'pose_cov': pose_cov,
             'cam_rel_mat': ivy.eye(4, batch_shape=[batch_size, num_frames], device=dev_str)[..., 0:3, :]}

    if ones:
        control_mean = ivy.zeros([batch_size, num_frames, 6], device=dev_str)
        control_cov = ivy.ones([batch_size, num_frames, 6, 6], device=dev_str)*1e-3
    else:
        control_mean = ivy.random_uniform(low=1e-3, high=1, shape=[batch_size, num_frames, 6], device=dev_str)
        control_cov = ivy.random_uniform(low=1e-3, high=1, shape=[batch_size, num_frames, 6, 6], device=dev_str)
    return ivy.Container({'img_meas': img_meas,
                      'control_mean': control_mean,
                      'control_cov': control_cov,
                      'agent_rel_mat': ivy.eye(4, batch_shape=[batch_size, num_frames],
                                                    device=dev_str)[..., 0:3, :]})


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
    esm(_get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels),
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
    ivy.seed(seed_value=0)
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
    for i in range(50):
        obs = _get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels, device)
        memory = esm(obs, memory, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                     image_dims=image_dims)
        memory_mean = memory['mean'].numpy()
        assert memory_mean.shape == tuple([batch_size, num_timesteps] + omni_img_dims + [3 + num_feature_channels])
        assert memory_mean[0, 0, 0, 0, 0] == 0.
        np.max(memory_mean)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    assert time_taken < 30.


def test_incremental_rotation(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    batch_size = 1
    num_timesteps = 1
    num_cams = 1
    num_feature_channels = 3
    image_dims = [3, 3]
    esm = ESM(omni_image_dims=[10, 20], smooth_mean=False)
    empty_memory = esm.empty_memory(batch_size, num_timesteps)
    empty_obs = _get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels, empty=True)
    rel_rot_vec_pose = ivy.array([[[0., 0., 0., 0., 0.1, 0.]]])
    empty_obs['control_mean'] = rel_rot_vec_pose
    empty_obs['agent_rel_mat'] = ivy_mech.rot_vec_pose_to_mat_pose(rel_rot_vec_pose)

    first_obs = _get_dummy_obs(batch_size, num_timesteps, num_cams, image_dims, num_feature_channels, ones=True)
    memory_1 = esm(first_obs, empty_memory, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                   image_dims=image_dims)
    memory_2 = esm(empty_obs, memory_1, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                   image_dims=image_dims)
    memory_3 = esm(empty_obs, memory_2, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                   image_dims=image_dims)

    assert not np.allclose(memory_1['mean'], memory_3['mean'])


def test_values(dev_str, call):
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    device = 'cpu'
    batch_size = 1
    num_timesteps = 1
    num_cams = 1
    num_feature_channels = 3
    image_dims = [128, 128]
    omni_img_dims = [180, 360]
    esm = ESM(omni_image_dims=omni_img_dims, device=device)
    memory = esm.empty_memory(batch_size, num_timesteps)
    this_dir = os.path.dirname(os.path.realpath(__file__))
    for i in range(2):
        obs = ivy.Container.from_disk_as_hdf5(os.path.join(this_dir, 'test_data/obs_{}.hdf5'.format(i)))
        memory = esm(obs, memory, batch_size=batch_size, num_timesteps=num_timesteps, num_cams=num_cams,
                     image_dims=image_dims)
        expected_mem = ivy.Container.from_disk_as_hdf5(os.path.join(this_dir, 'test_data/mem_{}.hdf5'.format(i)))
        assert np.allclose(memory['mean'], expected_mem['mean'], atol=1e-3)
        assert np.allclose(memory['var'], expected_mem['var'])
