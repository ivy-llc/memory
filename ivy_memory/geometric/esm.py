# global
import ivy
import ivy_mech
import ivy_vision

# local
from ivy_memory.geometric.containers import ESMObservation, ESMMemory

MIN_DENOMINATOR = 1e-12


# noinspection PyUnboundLocalVariable
class ESM(ivy.Module):

    def __init__(self, depth_prior=None, feat_prior=None, num_feature_channels=3, smooth_kernel_size=25,
                 threshold_var_factor=0.99, depth_limits=(1e-3, 10.), depth_var_limits=(1e-3, 1e4),
                 feat_var_limits=(1e-3, 1e4), omni_image_dims=(180, 360), smooth_mean=True, depth_buffer=True,
                 stateful=False, device='cpu'):
        # ToDo: make variance fully optional. If not specified,
        #  then do not compute and scatter during function call for better efficiency.

        # features
        self._feat_dim = num_feature_channels

        # hole priors
        if depth_prior is None:
            depth_prior = ivy.ones((1,), dev_str=device) * 1e-2
        if feat_prior is None:
            feat_prior = ivy.ones((num_feature_channels,), dev_str=device) * 0.5
        self._sphere_depth_prior_val = depth_prior
        self._sphere_feat_prior_val = feat_prior

        self._sphere_depth_prior = None
        self._sphere_feat_prior = None

        # value clipping
        self._min_depth = ivy.array(depth_limits[0], dev_str=device)
        self._max_depth = ivy.array(depth_limits[1], dev_str=device)

        # variance clipping
        ang_pix_var_limits = (1e-3, float(omni_image_dims[0]))
        self._min_ang_pix_var = ivy.array(ang_pix_var_limits[0], dev_str=device)
        self._min_depth_var = ivy.array(depth_var_limits[0], dev_str=device)
        self._min_feat_var = ivy.array(feat_var_limits[0], dev_str=device)

        self._ang_pix_prior_var_val = ivy.array(ang_pix_var_limits[1], dev_str=device)
        self._depth_prior_var_val = ivy.array(depth_var_limits[1], dev_str=device)
        self._feat_prior_var_val = ivy.array(feat_var_limits[1], dev_str=device)

        self._threshold_var_factor = ivy.array(threshold_var_factor, dev_str=device)
        self._ang_pix_var_threshold = ivy.array(ang_pix_var_limits[1] * self._threshold_var_factor, dev_str=device)
        self._depth_var_threshold = ivy.array(depth_var_limits[1] * self._threshold_var_factor, dev_str=device)
        self._feat_var_threshold = ivy.array(feat_var_limits[1] * self._threshold_var_factor, dev_str=device)

        # normalization
        self._depth_range = self._max_depth - self._min_depth

        self._min_log_depth_var = ivy.log(self._min_depth_var)
        self._min_log_feat_var = ivy.log(self._min_feat_var)

        max_log_depth_var = ivy.log(self._depth_prior_var_val)
        max_log_feat_var = ivy.log(self._feat_prior_var_val)

        self._log_depth_var_range = max_log_depth_var - self._min_log_depth_var
        self._log_feat_var_range = max_log_feat_var - self._min_log_feat_var

        # sphere image
        self._sphere_img_dims = list(omni_image_dims)
        self._pixels_per_degree = self._sphere_img_dims[0] / 180

        # image smoothing
        self._smooth_kernel_size = smooth_kernel_size
        self._smooth_mean = smooth_mean

        # rendering config
        self._with_depth_buffer = depth_buffer

        # memory
        self._stateful = stateful
        self._memory = None

        # variables
        ivy.Module.__init__(self, device)

    # Helpers #
    # --------#

    def _normalize_mean(self, mean):
        normed_depth = (mean[..., 0:1] - self._min_depth) / (self._depth_range + MIN_DENOMINATOR)
        normed_feat = mean[..., 3:]
        return ivy.concatenate((normed_depth, normed_feat), -1)

    def _normalize_var(self, var):
        normed_depth_var = (ivy.log(var[..., 0:1]) - self._min_log_depth_var) / \
                           (self._log_depth_var_range + MIN_DENOMINATOR)
        normed_feat_var = (ivy.log(var[..., 4:]) - self._min_log_feat_var) / \
                          (self._log_feat_var_range + MIN_DENOMINATOR)
        return ivy.concatenate((normed_depth_var, normed_feat_var), -1)

    @staticmethod
    def _fuse_measurements_with_uncertainty(measurements, measurement_uncertainties, axis):
        measurements_shape = measurements.shape
        batch_shape = measurements_shape[0:axis]
        num_batch_dims = len(batch_shape)
        num_measurements = measurements_shape[axis]

        # BS x 1 x RS
        sum_of_variances = ivy.reduce_sum(measurement_uncertainties, axis, keepdims=True)
        prod_of_variances = ivy.reduce_prod(measurement_uncertainties, axis, keepdims=True)

        # BS x 1 x RS
        new_variance = prod_of_variances / sum_of_variances

        # dim size list of BS x (dim-1) x RS
        batch_slices = [slice(None, None, None)] * num_batch_dims
        concat_lists = \
            [[measurement_uncertainties[tuple(batch_slices + [slice(0, i, None)])]]
             if i != 0 else [] + [measurement_uncertainties[tuple(batch_slices + [slice(i + 1, None, None)])]]
             for i in range(num_measurements)]
        partial_variances_list = [ivy.concatenate(concat_list, axis) for concat_list in concat_lists]

        # dim size list of BS x 1 x RS
        partial_prod_of_variances_list = \
            [ivy.reduce_prod(partial_variance, axis, True) for partial_variance in
             partial_variances_list]

        # BS x dim x RS
        partial_prod_of_variances = ivy.concatenate(partial_prod_of_variances_list, axis)

        # BS x 1 x RS
        new_mean = ivy.reduce_sum((partial_prod_of_variances * measurements) / sum_of_variances, axis, keepdims=True)

        # BS x 1 x RS, BS x 1 x RS
        return new_mean, new_variance

    # Projection Functions #
    # ---------------------#

    def _frame_to_omni_frame_projection(self, cam_rel_poses, cam_rel_mats, uniform_sphere_pixel_coords, cam_coords_f1,
                                        cam_feat_f1, rel_pose_covs, image_var_f1, holes_prior, holes_prior_var,
                                        batch_size, num_timesteps, num_cams, image_dims):
        """
        Project mean and variance values from frame 1 to omni frame 2, using scatter and quantize operations.
        :param cam_rel_poses: Relative pose of camera to agent *[batch_size, n, c, 6]*
        :param cam_rel_mats: Relative transformation matrix from camera to agent *[batch_size, n, c, 3, 4]*
        :param uniform_sphere_pixel_coords: Pixel coords *[batch_size, n, h, w, 3]*
        :param cam_coords_f1: Camera co-ordinates to project *[batch_size, n, c, h_in, w_in, 3]*
        :param cam_feat_f1: Mean feature values to project *[batch_size, n, c, h_in, w_in, f]*
        :param rel_pose_covs: Pose covariances *[batch_size, n, c, 6, 6]*
        :param image_var_f1: Angular pixel, radius and feature variance values to project *[batch_size, n, c, h_in, w_in, 3+f]*
        :param holes_prior: Prior values for holes which arise during the projection *[batch_size, n, h, w, 3+f]*
        :param holes_prior_var: Prior variance for holes which arise during the projection *[batch_size, n, h, w, 3+f]*
        :param batch_size: Size of batch
        :param num_timesteps: Number of timesteps
        :param num_cams: Number of cameras
        :param image_dims: Image dimensions
        :return: Projected mean *[batch_size, 1, h, w, 3+f]* and variance *[batch_size, 1, h, w, 3+f]*
        """

        # cam 1 to cam 2 coords

        cam_coords_f2 = ivy_vision.cam_to_cam_coords(
            ivy_mech.make_coordinates_homogeneous(
                cam_coords_f1, [batch_size, num_timesteps, num_cams] + image_dims), cam_rel_mats,
            [batch_size, num_timesteps, num_cams], image_dims)

        # cam 2 to sphere 2 coords

        sphere_coords_f2 = ivy_vision.cam_to_sphere_coords(cam_coords_f2)
        image_var_f2 = image_var_f1

        # angular pixel coords

        # B x N x C x H x W x 3
        angular_pixel_coords_f2 = \
            ivy_vision.sphere_to_angular_pixel_coords(sphere_coords_f2, self._pixels_per_degree)

        # constant feature projection

        # B x N x C x H x W x (3+F)
        projected_coords_f2 = ivy.concatenate([angular_pixel_coords_f2] + [cam_feat_f1], -1)

        # reshaping to fit quantization dimension requirements

        # B x N x (CxHxW) x (3+F)
        projected_coords_f2_flat = \
            ivy.reshape(projected_coords_f2,
                        [batch_size, num_timesteps, num_cams * image_dims[0] * image_dims[1], -1])

        # B x N x (CxHxW) x (3+F)
        image_var_f2_flat = ivy.reshape(image_var_f2,
                                        [batch_size, num_timesteps, num_cams * image_dims[0] * image_dims[1], -1])

        # quantized result from all scene cameras

        # B x N x OH x OW x (3+F)   # B x N x OH x OW x (3+F)
        return ivy_vision.quantize_to_image(
            pixel_coords=projected_coords_f2_flat[..., 0:2],
            final_image_dims=self._sphere_img_dims,
            feat=projected_coords_f2_flat[..., 2:],
            feat_prior=holes_prior,
            with_db=self._with_depth_buffer,
            pixel_coords_var=image_var_f2_flat[..., 0:2],
            feat_var=image_var_f2_flat[..., 2:],
            pixel_coords_prior_var=holes_prior_var[..., 0:2],
            feat_prior_var=holes_prior_var[..., 2:],
            var_threshold=self._var_threshold,
            uniform_pixel_coords=uniform_sphere_pixel_coords,
            batch_shape=(batch_size, num_timesteps),
            dev_str=self._dev_str)[0:2]

    def _omni_frame_to_omni_frame_projection(self, agent_rel_pose, agent_rel_mat, uniform_sphere_pixel_coords,
                                             sphere_pix_coords_f1, sphere_depth_f1, sphere_feat_f1, agent_rel_pose_cov,
                                             image_var_f1, holes_prior, holes_prior_var, batch_size):
        """
        Project mean and variance values from omni frame 1 to omni frame 2, using scatter and quantize operation.

        :param agent_rel_pose: Relative pose of agent to the previous step *[batch_size, 6]*
        :param agent_rel_mat: Relative transformation matrix of agent to the previous step
                                    *[batch_size, 3, 4]*
        :param uniform_sphere_pixel_coords: Pixel coords *[batch_size, h, w, 3]*
        :param sphere_pix_coords_f1: Pixel co-ordinates to project *[batch_size, h, w, 2]*
        :param sphere_depth_f1: Mean radius values to project *[batch_size, h, w, 1]*
        :param sphere_feat_f1: Mean feature values to project *[batch_size, h, w, f]*
        :param agent_rel_pose_cov: Agent relative pose covariance *[batch_size, 6, 6]*
        :param image_var_f1: Angular pixels, radius and feature variance values to project *[batch_size, h, w, 3+f]*
        :param holes_prior: Prior values for holes which arise during the projection *[batch_size, h, w, 1+f]*
        :param holes_prior_var: Prior variance for holes which arise during the projection
                                *[batch_size, h, w, 3+f]*
        :param batch_size: Size of batch
        :return: Projected mean *[batch_size, h, w, 3+f]* and variance *[batch_size, h, w, 3+f]*
        """

        # Frame 1 #
        # --------#

        # combined

        # B x OH x OW x 3
        angular_pixel_coords_f1 = ivy.concatenate((sphere_pix_coords_f1, sphere_depth_f1), -1)

        # sphere coords

        # B x OH x OW x 3
        sphere_coords_f1 = \
            ivy_vision.angular_pixel_to_sphere_coords(angular_pixel_coords_f1, self._pixels_per_degree)

        # Frame 2 #
        # --------#

        # sphere to sphere pixel projection

        sphere_coords_f2 = ivy_vision.sphere_to_sphere_coords(
            sphere_coords_f1, agent_rel_mat, [batch_size], self._sphere_img_dims)
        image_var_f2 = image_var_f1

        # to angular pixel coords

        # B x OH x OW x 3
        angular_pixel_coords_f2 = \
            ivy_vision.sphere_to_angular_pixel_coords(sphere_coords_f2, self._pixels_per_degree)

        # constant feature projection

        # B x OH x OW x (3+F)
        projected_coords_f2 = ivy.concatenate([angular_pixel_coords_f2] + [sphere_feat_f1], -1)

        # reshaping to fit quantization dimension requirements

        # B x (OHxOW) x (3+F)
        projected_coords_f2_flat = ivy.reshape(projected_coords_f2,
                                               [batch_size] + [self._sphere_img_dims[0] * self._sphere_img_dims[1]]
                                               + [3 + self._feat_dim])

        # B x (OHxOW) x (3+F)
        image_var_f2_flat = ivy.reshape(image_var_f2,
                                        [batch_size] + [self._sphere_img_dims[0] * self._sphere_img_dims[1]]
                                        + [3 + self._feat_dim])

        # quantize the projection

        # B x N x OH x OW x (3+F)   # B x N x OH x OW x (3+F)
        return ivy_vision.quantize_to_image(
            pixel_coords=projected_coords_f2_flat[..., 0:2],
            final_image_dims=self._sphere_img_dims,
            feat=projected_coords_f2_flat[..., 2:],
            feat_prior=holes_prior,
            with_db=self._with_depth_buffer,
            pixel_coords_var=image_var_f2_flat[..., 0:2],
            feat_var=image_var_f2_flat[..., 2:],
            pixel_coords_prior_var=holes_prior_var[..., 0:2],
            feat_prior_var=holes_prior_var[..., 2:],
            var_threshold=self._var_threshold[:, 0],
            uniform_pixel_coords=uniform_sphere_pixel_coords,
            batch_shape=(batch_size,),
            dev_str=self._dev_str)[0:2]

    # Measurement #
    # ------------#

    def _convert_images_to_omni_observations(self, measurements, uniform_sphere_pixel_coords,
                                             holes_prior, batch_size, num_timesteps, num_cams, image_dims):
        """
        Convert image to omni-directional measurements

        :param measurements: perspective captured images and relative poses container
        :param uniform_sphere_pixel_coords: Uniform  sphere pixel coords *[batch_size, num_timesteps, oh, ow, 3]*
        :param holes_prior: Prior for quantization holes *[batch_size, num_timesteps, oh, ow, 1+f]*
        :param batch_size: Size of batch
        :param num_timesteps: Number of frames
        :param num_cams: Number of cameras
        :param image_dims: Image dimensions
        :return: *[batch_size, n, oh, ow, 3+f]*    *[batch_size, n, oh, ow, 3+f]*
        """

        # coords from all scene cameras wrt world

        images_list = list()
        images_var_list = list()
        cam_rel_poses_list = list()
        cam_rel_poses_cov_list = list()
        cam_rel_mats_list = list()
        validity_mask_list = list()
        for key, item in measurements.to_iterator(leaf_keys_only=True):
            if key == 'img_mean':
                # B x N x 1 x H x W x (3+f)
                images_list.append(ivy.expand_dims(item, 2))
            elif key == 'img_var':
                # B x N x 1 x H x W x (3+f)
                images_var_list.append(ivy.expand_dims(item, 2))
            elif key == 'pose_mean':
                # B x N x 1 x 6
                cam_rel_poses_list.append(ivy.expand_dims(item, 2))
            elif key == 'pose_cov':
                # B x N x 1 x 6 x 6
                cam_rel_poses_cov_list.append(ivy.expand_dims(item, 2))
            elif key == 'cam_rel_mat':
                # B x N x 1 x 3 x 4
                cam_rel_mats_list.append(ivy.expand_dims(item, 2))
            elif key == 'validity_mask':
                validity_mask_list.append(ivy.expand_dims(item, 2))
            else:
                raise Exception('Invalid image key: {}'.format(key))

        # B x N x C x H x W x (3+f)
        images = ivy.concatenate(images_list, 2)

        # B x N x C x H x W x (3+f)
        var_to_project = ivy.concatenate(images_var_list, 2)

        # B x N x C x 6
        cam_to_cam_poses = ivy.concatenate(cam_rel_poses_list, 2)

        # B x N x C x 3 x 4
        cam_to_cam_mats = ivy.concatenate(cam_rel_mats_list, 2)

        # B x N x C x 6 x 6
        cam_to_cam_pose_covs = ivy.concatenate(cam_rel_poses_cov_list, 2)

        # B x N x C x 1
        validity_masks = ivy.concatenate(validity_mask_list, 2) > 0

        # B x N x OH x OW x (3+f)
        holes_prior_var = ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [3 + self._feat_dim],
                                   dev_str=self._dev_str) * 1e12

        # reset invalid regions to prior

        # B x N x C x H x W x (3+f)
        images = ivy.where(validity_masks, images,
                           ivy.concatenate((images[..., 0:2], ivy.zeros_like(images[..., 2:], dev_str=self._dev_str)), -1))

        # B x N x C x H x W x (3+f)
        var_to_project = ivy.where(validity_masks, var_to_project,
                                   ivy.ones_like(var_to_project, dev_str=self._dev_str) * 1e12)

        # B x N x OH x OW x (3+f)    # B x N x OH x OW x (3+f)
        return self._frame_to_omni_frame_projection(
            cam_to_cam_poses, cam_to_cam_mats, uniform_sphere_pixel_coords, images[..., 0:3], images[..., 3:],
            cam_to_cam_pose_covs, var_to_project, holes_prior, holes_prior_var, batch_size, num_timesteps, num_cams,
            image_dims)

    # Kalman Filter #
    # --------------#

    def _kalman_filter_on_measurement_sequence(self, prev_fused_val, prev_fused_variance, hole_prior,
                                               hole_prior_var, meas, meas_vars, uniform_sphere_pixel_coords,
                                               agent_rel_poses, agent_rel_pose_covs, agent_rel_mats,
                                               batch_size, num_timesteps):
        """
        Perform kalman filter on measurement sequence

        :param prev_fused_val: Fused value from previous timestamp *[batch_size, oh, ow, (3+f)]*
        :param prev_fused_variance: Fused variance from previous timestamp *[batch_size, oh, ow, (3+f)]*
        :param hole_prior: Prior for holes in quantization *[batch_size, oh, ow, (1+f)]*
        :param hole_prior_var: Prior variance for holes in quantization *[batch_size, oh, ow, (3+f)]*
        :param meas: Measurements *[batch_size, num_timesteps, oh, ow, (3+f)]*
        :param meas_vars: Measurement variances *[batch_size, num_timesteps, oh, ow, (3+f)]*
        :param uniform_sphere_pixel_coords: Uniform sphere pixel co-ordinates *[batch_size, oh, ow, 3]*
        :param agent_rel_poses: Relative poses of agents to the previous step *[batch_size, num_timesteps, 6]*
        :param agent_rel_pose_covs: Agent relative pose covariances *[batch_size, num_timesteps, 6, 6]*
        :param agent_rel_mats: Relative transformations matrix of agents to the previous step
                                *[batch_size, num_timesteps, 3, 4]*
        :param batch_size: Size of batch
        :param num_timesteps: Number of frames
        :return: list of *[batch_size, oh, ow, (3+f)]*,    list of *[batch_size, oh, ow, (3+f)]*
        """

        fused_list = list()
        fused_variances_list = list()

        for i in range(num_timesteps):
            # project prior from previous frame #
            # ----------------------------------#

            # B x OH x OW x (3+F)
            prev_prior = prev_fused_val
            prev_prior_variance = prev_fused_variance

            # B x 3 x 4
            agent_rel_mat = agent_rel_mats[:, i]

            # B x 6
            agent_rel_pose = agent_rel_poses[:, i]

            # B x 6 x 6
            agent_rel_pose_cov = agent_rel_pose_covs[:, i]

            # B x OH x OW x (3+F)   B x OH x OW x (3+F)
            fused_projected, fused_projected_variance = self._omni_frame_to_omni_frame_projection(
                agent_rel_pose, agent_rel_mat, uniform_sphere_pixel_coords, prev_prior[..., 0:2], prev_prior[..., 2:3],
                prev_prior[..., 3:], agent_rel_pose_cov, prev_prior_variance, hole_prior, hole_prior_var, batch_size)

            # reset prior

            # B x OH x OW x (3+F)
            prior = fused_projected
            prior_var = fused_projected_variance

            # per-pixel fusion with measurements #
            # -----------------------------------#

            # extract slice for frame

            # B x OH x OW x (3+F)
            measurement = meas[:, i]
            measurement_variance = meas_vars[:, i]

            # fuse prior and measurement

            # B x 2 x OH x OW x (3+F)
            prior_and_meas = ivy.concatenate((ivy.expand_dims(prior, 1),
                                              ivy.expand_dims(measurement, 1)), 1)
            prior_and_meas_variance = ivy.concatenate((ivy.expand_dims(prior_var, 1),
                                                       ivy.expand_dims(measurement_variance, 1)), 1)

            # B x OH x OW x (3+F)
            low_var_mask = ivy.reduce_sum(ivy.cast(
                prior_and_meas_variance < ivy.expand_dims(hole_prior_var, 1) * self._threshold_var_factor,
                'int32'), 1) > 0

            # B x 1 x OH x OW x (3+F)    B x 1 x OH x OW x (3+F)
            # ToDo: handle this properly once re-implemented with a single scatter operation only
            #  currently depth values are fused even if these are clearly far apart
            fused_val_unsmoothed, fused_variance_unsmoothed = \
                self._fuse_measurements_with_uncertainty(prior_and_meas, prior_and_meas_variance, 1)

            # B x OH x OW x (3+F)
            # This prevents accumulating certainty from duplicate re-projections from prior measurements
            fused_variance_unsmoothed = ivy.where(low_var_mask, fused_variance_unsmoothed[:, 0], hole_prior_var)

            # B x OH x OW x (3+F)
            fused_val = fused_val_unsmoothed[:, 0]
            fused_variance = fused_variance_unsmoothed
            low_var_mask = fused_variance < hole_prior_var

            # B x OH x OW x (3+F)    B x OH x OW x (3+F)
            fused_val, fused_variance = self.smooth(
                fused_val, fused_variance, low_var_mask, self._smooth_mean, self._smooth_kernel_size,
                True, True, batch_size)

            # append to list for returning

            # B x OH x OW x (3+F)
            fused_list.append(fused_val)

            # B x OH x OW x (3+F)
            fused_variances_list.append(fused_variance)

            # update for next time step
            prev_fused_val = fused_val
            prev_fused_variance = fused_variance

        # list of *[batch_size, oh, ow, (3+f)]*,    list of *[batch_size, oh, ow, (3+f)]*
        return fused_list, fused_variances_list

    # Public Functions #
    # -----------------#

    def empty_memory(self, batch_size, timesteps):
        uniform_pixel_coords = \
            ivy_vision.create_uniform_pixel_coords_image(self._sphere_img_dims, [batch_size, timesteps],
                                                         dev_str=self._dev_str)[..., 0:2]
        empty_memory = {
            'mean': ivy.concatenate(
                [uniform_pixel_coords] +
                [ivy.ones([batch_size, timesteps] + self._sphere_img_dims + [1], dev_str=self._dev_str)
                 * self._sphere_depth_prior_val] +
                [ivy.ones([batch_size, timesteps] + self._sphere_img_dims + [self._feat_dim], dev_str=self._dev_str)
                 * self._sphere_feat_prior_val], -1),
            'var': ivy.concatenate(
                [ivy.ones([batch_size, timesteps] + self._sphere_img_dims + [2], dev_str=self._dev_str)
                 * self._ang_pix_prior_var_val] +
                [ivy.ones([batch_size, timesteps] + self._sphere_img_dims + [1], dev_str=self._dev_str)
                 * self._depth_prior_var_val] +
                [ivy.ones([batch_size, timesteps] + self._sphere_img_dims + [self._feat_dim], dev_str=self._dev_str)
                 * self._feat_prior_var_val], -1)
        }
        return ESMMemory(**empty_memory)

    def smooth(self, fused_val, fused_variance, low_var_mask, smooth_mean,
               smooth_kernel_size, variance_mode, fix_low_var_pixels, batch_size):
        variance_mode = True
        var_for_smoothing = fused_variance

        # image smoothing
        if smooth_mean:

            # pad borders
            pad_size = int(smooth_kernel_size / 2)
            fused_val_pad = ivy_vision.pad_omni_image(fused_val, pad_size, self._sphere_img_dims)
            fused_variance_pad = ivy_vision.pad_omni_image(var_for_smoothing, pad_size, self._sphere_img_dims)
            expanded_sphere_img_dims = [item + 2 * pad_size for item in self._sphere_img_dims]

            # reshape for dilation and erosion
            fused_val_pad_flat = ivy.reshape(fused_val_pad[..., 2:], [batch_size] + expanded_sphere_img_dims +
                                             [1 + self._feat_dim])
            fused_var_pad_flat = ivy.reshape(fused_variance_pad[..., 2:], [batch_size] + expanded_sphere_img_dims +
                                             [1 + self._feat_dim])

            if variance_mode:
                smoothed_fused_val_flat, smoothed_fused_var_flat = \
                    ivy_vision.smooth_image_fom_var_image(fused_val_pad_flat, fused_var_pad_flat, smooth_kernel_size,
                                                          ivy.array([1.] * (1 + self._feat_dim), dev_str=self._dev_str))
            else:
                smoothed_fused_val_flat, smoothed_fused_var_flat = \
                    ivy_vision.weighted_image_smooth(fused_val_pad_flat, 1 - fused_var_pad_flat, smooth_kernel_size)

            # reshape to image dims
            smoothed_fused_val = ivy.reshape(smoothed_fused_val_flat, [batch_size] + self._sphere_img_dims +
                                             [1 + self._feat_dim])
            smoothed_fused_val = ivy.concatenate((fused_val[..., 0:2], smoothed_fused_val), -1)

            # replace temporary zeros with their prior values
            # This ensures that the smoothing operation only changes the values for regions of high variance
            if fix_low_var_pixels:
                fused_val = ivy.where(low_var_mask, fused_val, smoothed_fused_val)
            else:
                fused_val = smoothed_fused_val

        return fused_val, fused_variance

    # Main call method #
    # -----------------#

    def _forward(self, obs: ESMObservation, memory: ESMMemory = None, batch_size=None, num_timesteps=None,
                 num_cams=None, image_dims=None):
        """
        Perform ESM update step.

        :param obs: Observations
        :type obs: ESMObservation
        :param memory: Memory from the previous time-step, uses internal parameter if None.
        :type memory: ESMMemory, optional.
        :param batch_size: Size of batch, inferred from inputs if None.
        :type batch_size: int, optional
        :param num_timesteps: Number of timesteps, inferred from inputs if None.
        :type num_timesteps: int, optional
        :param num_cams: Number of cameras, inferred from inputs if None.
        :type num_cams: int, optional
        :param image_dims: Image dimensions of captured images, inferred from inputs if None.
        :type image_dims: sequence of ints, optional
        :return: New memory of type ESMMemory
        """

        # get shapes
        img_meas = (next(iter(obs.img_meas.values()))).img_mean
        if batch_size is None:
            batch_size = int(img_meas.shape[0])
        if num_timesteps is None:
            num_timesteps = int(img_meas.shape[1])
        if num_cams is None:
            num_cams = len(obs.img_meas.values())
        if image_dims is None:
            image_dims = list(img_meas.shape[2:4])

        # get only previous memory

        # extract from memory #
        # --------------------#

        if memory:
            prev_mem = memory[:, -1]
        elif self._stateful and self._memory is not None:
            prev_mem = self._memory[:, -1]
        else:
            prev_mem = self.empty_memory(batch_size, 1)[:, -1]

        # holes prior #
        # ------------#

        # B x N x OH x OW x 1
        self._sphere_depth_prior = \
            ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [1], dev_str=self._dev_str) * \
            self._sphere_depth_prior_val

        # B x N x OH x OW x F
        self._sphere_feat_prior = \
            ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [self._feat_dim], dev_str=self._dev_str) * \
            self._sphere_feat_prior_val

        # B x N x OH x OW x (1+F)
        holes_prior = ivy.concatenate([self._sphere_depth_prior] + [self._sphere_feat_prior], -1)

        # holes prior variances #
        # ----------------------#

        # B x N x OH x OW x 2
        sphere_ang_pix_prior_var = \
            ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [2], dev_str=self._dev_str) * self._ang_pix_prior_var_val

        # B x N x OH x OW x 1
        sphere_depth_prior_var = \
            ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [1], dev_str=self._dev_str) * self._depth_prior_var_val

        # B x N x OH x OW x F
        sphere_feat_prior_var = \
            ivy.ones([batch_size, num_timesteps] + self._sphere_img_dims + [self._feat_dim], dev_str=self._dev_str) * \
            self._feat_prior_var_val

        # B x N x OH x OW x (3+F)
        holes_prior_var = ivy.concatenate([sphere_ang_pix_prior_var] +
                                          [sphere_depth_prior_var] +
                                          [sphere_feat_prior_var], -1)

        # variance threshold #
        # -------------------#

        # B x N x (3+F) x 1
        var_threshold_min = ivy.tile(
            ivy.reshape(ivy.stack([self._min_ang_pix_var]*2 +
                                  [self._min_depth_var] +
                                  [self._min_feat_var] * self._feat_dim),
                        [1, 1, 3 + self._feat_dim, 1]), [batch_size, num_timesteps, 1, 1])
        var_threshold_max = ivy.tile(ivy.reshape(ivy.stack([self._ang_pix_var_threshold] * 2 +
                                                           [self._depth_var_threshold] +
                                                           [self._feat_var_threshold] * self._feat_dim),
                                                 [1, 1, 3 + self._feat_dim, 1]), [batch_size, num_timesteps, 1, 1])
        self._var_threshold = ivy.concatenate((var_threshold_min, var_threshold_max), -1)

        # measurements #
        # -------------#

        # B x N x OH x OW x 3
        uniform_sphere_pixel_coords = ivy_vision.create_uniform_pixel_coords_image(
            self._sphere_img_dims, (batch_size, num_timesteps), dev_str=self._dev_str)

        # B x N x OH x OW x (3+F),    B x N x OH x OW x (3+F)
        meas_means, meas_vars = self._convert_images_to_omni_observations(
            obs.img_meas, uniform_sphere_pixel_coords, holes_prior, batch_size, num_timesteps, num_cams, image_dims)

        # filtering #
        # ----------#

        # list of B x OH x OW x (3+F),    list of B x OH x OW x (3+F)
        fused_measurements_list, fused_variances_list = \
            self._kalman_filter_on_measurement_sequence(
                prev_mem.mean, prev_mem.var, holes_prior[:, 0], holes_prior_var[:, 0], meas_means, meas_vars,
                uniform_sphere_pixel_coords[:, 0], obs.control_mean, obs.control_cov, obs.agent_rel_mat,
                batch_size, num_timesteps)

        # new variance #
        # -------------#

        # B x N x OH x OW x (3+F)
        fused_variance = ivy.concatenate([ivy.expand_dims(item, 1) for item in fused_variances_list], 1)

        # variance clipping

        # B x N x OH x OW x 1
        fused_depth_variance = ivy.clip(fused_variance[..., 0:1], self._min_depth_var, self._depth_prior_var_val)

        # B x N x OH x OW x 3
        fused_feat_variance = ivy.clip(fused_variance[..., 1:], self._min_feat_var, self._feat_prior_var_val)

        # B x N x OH x OW x (3+F)
        fused_variance = ivy.concatenate([fused_depth_variance] + [fused_feat_variance], -1)

        # new mean #
        # ---------#

        # B x N x OH x OW x (3+F)
        fused_measurement = ivy.concatenate([ivy.expand_dims(item, 1) for item in fused_measurements_list], 1)

        # value clipping

        # B x N x OH x OW x 2
        fused_pixel_coords = fused_measurement[..., 0:2]

        # B x N x OH x OW x 1
        fused_depth = ivy.clip(fused_measurement[..., 2:3], self._min_depth, self._max_depth)

        # B x N x OH x OW x 3
        fused_feat = fused_measurement[..., 3:]

        # B x N x OH x OW x (3+F)
        fused_measurement = ivy.concatenate([fused_pixel_coords] + [fused_depth] + [fused_feat], -1)

        # update memory #
        # --------------#

        # B x N x OH x OW x (3+F),    B x N x OH x OW x (3+F)
        self._memory = ESMMemory(mean=fused_measurement, var=fused_variance)

        # return #
        # -------#

        return self._memory
