"""
Collection of containers for handling ESM input and output data
"""

# global
import ivy
import ivy_mech
from typing import Dict
from ivy.core.container import Container


def _pad_to_batch_n_time_dims(data, expected_dims):
    # ToDo: remove this once ESM supports arbitrary batch dimensions
    found_dims = len(data.shape)
    if found_dims == expected_dims:
        return data
    elif found_dims < expected_dims:
        return ivy.reshape(data, [1]*(expected_dims - found_dims) + list(data.shape))
    else:
        raise Exception('found more dims {} than expected {}'.format(found_dims, expected_dims))


# noinspection PyMissingConstructor
class ESMCamMeasurement(Container):

    def __init__(self,
                 img_mean: ivy.Array,
                 cam_rel_mat: ivy.Array = None,
                 img_var: ivy.Array = None,
                 validity_mask: ivy.Array = None,
                 pose_mean: ivy.Array = None,
                 pose_cov: ivy.Array = None,
                 dev_str: str = None):
        """
        Create esm image measurement container

        :param img_mean: Camera-relative co-ordinates and image features
                            *[batch_size, timesteps, height, width, 3 + feat]*
        :type: img_mean: array
        :param cam_rel_mat: The pose of the camera relative to the current agent pose. Default is identity matrix
                            *[batch_size, timesteps, 3, 4]*
        :type cam_rel_mat: array, optional
        :param img_var: Image depth and feature variance values, assumed all zero if None.
                        *[batch_size, timesteps, height, width, 1 + feat]*
        :type: img_var: array, optional
        :param validity_mask: Validity mask, for which pixels should be considered. Assumed all valid if None
                                *[batch_size, timesteps, height, width, 1]*
        :type validity_mask: array, optional
        :param pose_mean: The pose of the camera relative to the current agent pose, in rotation vector pose form.
                            Inferred from cam_rel_mat if None. *[batch_size, timesteps, 6]*
        :type pose_mean: array, optional
        :param pose_cov: The convariance of the camera relative pose, in rotation vector form. Assumed all zero if None.
                            *[batch_size, timesteps, 6, 6]*
        :type pose_cov: array, optional
        :param dev_str: Device string to use, default is to use img_mean.
        :type dev_str: str
        """
        if dev_str is None:
            dev_str = ivy.dev_str(img_mean)
        img_mean = _pad_to_batch_n_time_dims(img_mean, 5)
        cont = dict()
        cont['img_mean'] = img_mean
        if cam_rel_mat is None:
            cam_rel_mat = ivy.identity(4, batch_shape=img_mean.shape[0:2], dev_str=dev_str)[..., 0:3, :]
        else:
            cam_rel_mat = _pad_to_batch_n_time_dims(cam_rel_mat, 4)
        cont['cam_rel_mat'] = cam_rel_mat
        if img_var is None:
            img_var = ivy.zeros_like(img_mean, dev_str=dev_str)
        else:
            img_var = _pad_to_batch_n_time_dims(img_var, 5)
        cont['img_var'] = img_var
        if validity_mask is None:
            validity_mask = ivy.ones_like(img_mean[..., 0:1], dev_str=dev_str)
        else:
            validity_mask = _pad_to_batch_n_time_dims(validity_mask, 5)
        cont['validity_mask'] = validity_mask
        if pose_mean is None:
            pose_mean = ivy_mech.mat_pose_to_rot_vec_pose(cam_rel_mat)
        else:
            pose_mean = _pad_to_batch_n_time_dims(pose_mean, 3)
        cont['pose_mean'] = pose_mean
        if pose_cov is None:
            pose_cov = ivy.tile(ivy.expand_dims(ivy.zeros_like(pose_mean, dev_str=dev_str), -1), (1, 1, 1, 6))
        else:
            pose_cov = _pad_to_batch_n_time_dims(pose_cov, 4)
        cont['pose_cov'] = pose_cov
        Container.__init__(self, cont)


# noinspection PyMissingConstructor
class ESMObservation(Container):

    def __init__(self,
                 img_meas: Dict[str, ESMCamMeasurement],
                 agent_rel_mat: ivy.Array,
                 control_mean: ivy.Array = None,
                 control_cov: ivy.Array = None,
                 dev_str: str = None):
        """
        Create esm observation container

        :param img_meas: dict of ESMImageMeasurement objects, with keys for camera names.
        :type: img_meas: Ivy container
        :param agent_rel_mat: The pose of the agent relative to the previous pose, in matrix form
                                *[batch_size, timesteps, 3, 4]*.
        :type agent_rel_mat: array
        :param control_mean: The pose of the agent relative to the previous pose, in rotation vector pose form.
                                Inferred from agent_rel_mat if None. *[batch_size, timesteps, 6]*
        :type control_mean: array, optional
        :param control_cov: The convariance of the agent relative pose, in rotation vector form.
                             Assumed all zero if None. *[batch_size, timesteps, 6, 6]*.
        :type control_cov: array, optional
        :param dev_str: Device string to use, default is to use img_mean.
        :type dev_str: str
        """
        cont = dict()
        if dev_str is None:
            dev_str = ivy.dev_str(agent_rel_mat)
        cont['img_meas'] = Container(img_meas)
        agent_rel_mat = _pad_to_batch_n_time_dims(agent_rel_mat, 4)
        cont['agent_rel_mat'] = agent_rel_mat
        if control_mean is None:
            control_mean = ivy_mech.mat_pose_to_rot_vec_pose(agent_rel_mat)
        else:
            control_mean = _pad_to_batch_n_time_dims(control_mean, 3)
        cont['control_mean'] = control_mean
        if control_cov is None:
            control_cov = ivy.tile(ivy.expand_dims(ivy.zeros_like(control_mean, dev_str=dev_str), -1), (1, 1, 1, 6))
        else:
            control_cov = _pad_to_batch_n_time_dims(control_cov, 4)
        cont['control_cov'] = control_cov
        Container.__init__(self, cont)


# noinspection PyMissingConstructor
class ESMMemory(Container):

    def __init__(self,
                 mean: ivy.Array,
                 var: ivy.Array = None,
                 dev_str: str = None):
        """
        Create esm memory container

        :param mean: The ESM memory feature values *[batch_size, timesteps, omni_height, omni_width, 2 + feat]*
        :type: mean: array
        :param var: The ESM memory feature variance values. All assumed zero if None.
                        *[batch_size, timesteps, omni_height, omni_width, feat]*
        :type: var: array, optional
        :param dev_str: Device string to use, default is to use img_mean.
        :type dev_str: str
        """
        cont = dict()
        mean = _pad_to_batch_n_time_dims(mean, 5)
        cont['mean'] = mean
        if var is None:
            var = ivy.zeros_like(mean, dev_str=dev_str)
        else:
            var = _pad_to_batch_n_time_dims(var, 5)
        cont['var'] = var
        Container.__init__(self, cont)
