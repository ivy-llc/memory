# global
import os
import ivy
import argparse
import ivy_mech
import ivy_vision
import numpy as np
from ivy_demo_utils.ivy_scene.scene_utils import SimCam, BaseSimulator

# local
import ivy_memory as ivy_mem
from ivy_memory.geometric.containers import ESMCamMeasurement, ESMObservation


# Helpers #
# --------#


def _add_image_border(img):
    img[0:2] = 0
    img[-2:] = 0
    img[:, 0:2] = 0
    img[:, -2:] = 0
    return img


def _add_title(img, height, width, t, text, offset):
    import cv2

    img[0:height, -width:] = 0
    img[t : height - t, -width + t : -t] = 255
    cv2.putText(
        img,
        text,
        (img.shape[1] - offset, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        tuple([0.0] * 3),
        1,
    )
    return img.copy()


# Classes for offline demos #
# --------------------------#


class OfflineDrone:
    def __init__(self, cam):
        self._inv_ext_mat_homo = ivy.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.cam = cam
        self._time = 0

    def measure_incremental_mat(self):
        rel_mat = ivy.array(
            np.load(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "esm_no_sim/agent_rel_mat_{}.npy".format(str(self._time).zfill(2)),
                )
            )
        )
        self._time += 1
        return rel_mat


class OfflineDroneCam:
    def __init__(self):
        self.inv_calib_mat = ivy.array(
            [
                [-0.015625, -0.0, 0.9921875],
                [-0.0, -0.015625, 0.9921875],
                [0.0, 0.0, 1.0],
            ]
        )
        self._time = 0

    def cap(self):
        rgb = ivy.array(
            np.load(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "esm_no_sim/rgb_{}.npy".format(str(self._time).zfill(2)),
                )
            )
        )
        depth = ivy.array(
            np.load(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "esm_no_sim/depth_{}.npy".format(str(self._time).zfill(2)),
                )
            )
        )
        self._time += 1
        return depth, rgb

    @property
    def mat_rel_to_drone(self):
        return ivy.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15]]
        )


# Classes for live demos #
# -----------------------#


class Drone:
    def __init__(self, pyrep_handle, cam):
        self._handle = pyrep_handle
        self._inv_ext_mat_homo = ivy_mech.make_transformation_homogeneous(
            ivy.array(self._handle.get_matrix()[0:3].tolist())
        )
        self.cam = cam

    def measure_incremental_mat(self):
        inv_ext_mat = ivy.array(self._handle.get_matrix()[0:3].tolist())
        inv_ext_mat_homo = ivy_mech.make_transformation_homogeneous(inv_ext_mat)
        ext_mat_homo = ivy.inv(inv_ext_mat_homo)
        ext_mat = ext_mat_homo[0:3, :]
        rel_mat = ivy.matmul(ext_mat, self._inv_ext_mat_homo)
        self._inv_ext_mat_homo = inv_ext_mat_homo
        return rel_mat


class DroneCam(SimCam):
    def __init__(self, pyrep_handle):
        super().__init__(pyrep_handle)

    @property
    def mat_rel_to_drone(self):
        return ivy.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15]]
        )


class Simulator(BaseSimulator):
    def __init__(self, interactive, try_use_sim):
        super().__init__(interactive, try_use_sim)

        # initialize scene
        if self.with_pyrep:
            self._spherical_vision_sensor.remove()
            for i in range(1, 6):
                self._vision_sensors[i].remove()
                self._vision_sensor_bodies[i].remove()
                [item.remove() for item in self._vision_sensor_rays[i]]
            [item.remove() for item in self._vision_sensor_rays[0]]
            drone_start_pos = np.array([0, 0, 1.5])
            self._drone.set_position(drone_start_pos)
            self._default_camera.set_position(np.array([-1.218, 0.710, 3.026]))
            self._default_camera.set_orientation(np.array([2.642, 0.596, -0.800]))

            # make vision sensor child of drone
            self._vision_sensors[0]
            vision_sensor_body = self._vision_sensor_bodies[0]
            vision_sensor_body.set_quaternion([0.5, 0.5, 0.5, 0.5])
            vision_sensor_body.set_position(self._drone.get_position())
            vision_sensor_body.set_position([0.0, 0.0, 0.15], vision_sensor_body)
            vision_sensor_body.set_parent(self._drone)

            # public drone
            self.drone = Drone(self._drone, DroneCam(self._vision_sensors[0]))

            # wait for user input
            self._user_prompt(
                "\nInitialized scene with a drone in the centre.\n\n"
                "You can click on the drone,"
                "then select the box icon with four arrows in the top panel "
                "of the simulator, "
                "and then drag the drone around dynamically.\n"
                "Starting to drag and then holding ctrl allows you to also "
                "drag the camera up and down. \n\n"
                "This demo enables you to capture 10 different images "
                "from the drone forward facing camera, "
                "and render the first 10 point cloud representations of the "
                "ESM memory in an open3D visualizer.\n\n"
                "Both visualizers can be translated and rotated by clicking "
                "either the left mouse button or the wheel, "
                "and then dragging the mouse.\n"
                "Scrolling the mouse wheel zooms the view in and out.\n\n"
                "Both visualizers can be rotated and zoomed by clicking "
                "either the left mouse button or the wheel, "
                "and then dragging with the mouse.\n\n"
                "Press enter in the terminal to start the demo.\n\n"
            )

        else:
            self.drone = OfflineDrone(OfflineDroneCam())

            # message
            print(
                "\nInitialized dummy scene with a drone in the centre."
                "\nClose the visualization window to start the demo.\n"
            )


def main(interactive=True, try_use_sim=True, f=None, fw=None):
    # setup backend
    fw = ivy.choose_random_backend(excluded=["numpy", "jax"]) if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.with_backend(backend=fw) if f is None else f

    # simulator and drone
    sim = Simulator(interactive, try_use_sim)
    drone = sim.drone

    # ESM
    esm = ivy_mem.ESM()
    esm_mem = esm.empty_memory(1, 1)

    # demo loop
    for _ in range(1000 if interactive and sim.with_pyrep else 100):
        # log iteration
        print("timestep {}".format(_))

        # acquire image measurements
        depth, rgb = drone.cam.cap()

        # convert to ESM format
        ds_pix_coords = ivy_vision.depth_to_ds_pixel_coords(depth)
        cam_coords = ivy_vision.ds_pixel_to_cam_coords(
            ds_pix_coords, drone.cam.inv_calib_mat
        )[..., 0:3]
        img_mean = ivy.concat((cam_coords, rgb), axis=-1)

        # acquire pose measurements
        cam_rel_mat = drone.cam.mat_rel_to_drone
        agent_rel_mat = ivy.array(drone.measure_incremental_mat())

        # single esm camera measurement
        esm_cam_meas = ESMCamMeasurement(img_mean=img_mean, cam_rel_mat=cam_rel_mat)

        # total esm observation
        obs = ESMObservation(
            img_meas={"cam0": esm_cam_meas}, agent_rel_mat=agent_rel_mat
        )

        esm_mem = esm(obs, esm_mem)

        # update esm visualization
        if not interactive:
            continue
        import cv2

        rgb_img = _add_image_border(cv2.resize(ivy.to_numpy(rgb).copy(), (180, 180)))
        rgb_img = _add_title(rgb_img, 25, 75, 2, "raw rgb", 70)
        depth_img = _add_image_border(
            cv2.resize(
                np.clip(np.tile(ivy.to_numpy(depth), (1, 1, 3)) / 3, 0, 1).copy(),
                (180, 180),
            )
        )
        depth_img = _add_title(depth_img, 25, 90, 2, "raw depth", 85)
        raw_img_concatted = np.concatenate((rgb_img, depth_img), 0)
        esm_feat = _add_image_border(
            np.clip(ivy.to_numpy(esm_mem.mean[0, 0, ..., 3:]), 0, 1).copy()
        )
        esm_feat = _add_title(esm_feat, 25, 80, 2, "esm rgb", 75)
        esm_depth = _add_image_border(
            np.clip(
                np.tile(ivy.to_numpy(esm_mem.mean[0, 0, ..., 2:3]) / 3, (1, 1, 3)), 0, 1
            ).copy()
        )
        esm_depth = _add_title(esm_depth, 25, 95, 2, "esm depth", 90)
        esm_img_concatted = np.concatenate((esm_feat, esm_depth), 0)
        img_to_show = np.clip(
            np.concatenate((raw_img_concatted, esm_img_concatted), 1), 0, 1
        )
        import matplotlib.pyplot as plt

        plt.imshow(img_to_show)
        plt.show(block=False)
        plt.pause(0.001)

    # end of demo
    sim.close()
    ivy.previous_backend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="whether to run the demo in non-interactive mode.",
    )
    parser.add_argument(
        "--no_sim",
        action="store_true",
        help="whether to run the demo without attempt to use the PyRep simulator.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="which backend to use. Chooses a random backend if unspecified.",
    )
    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    f = None if fw is None else ivy.with_backend(backend=fw)
    main(not parsed_args.non_interactive, not parsed_args.no_sim, f, fw)
