from dataclasses import dataclass
import pickle
from typing import List
import imageio
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2
from config import Config
from datasets.colmap import Parser
from metric3d.mono.model.monodepth_model import get_configured_monodepth_model
from metric3d.mono.utils.do_test import get_prediction, transform_test_data_scalecano
from metric3d.mono.utils.running import load_ckpt

try:
    from mmcv.utils import Config as Metric3dConfig
except ImportError:
    from mmengine import Config as Metric3dConfig
from tqdm import tqdm


@dataclass
class Metric3dModel:
    cfg: Metric3dConfig
    model: torch.nn.Module

    @staticmethod
    def load(config: Config, device: str) -> "Metric3dModel":
        if config.metric3d_config is None:
            raise ValueError("Metric3d config path is not provided.")
        if config.metric3d_weights is None:
            raise ValueError("Metric3d weights path is not provided.")

        cfg = Metric3dConfig.fromfile(config.metric3d_config)
        model = get_configured_monodepth_model(
            cfg,
        )
        model, _, _, _ = load_ckpt(config.metric3d_weights, model, strict_match=False)
        model.eval()
        model.to(device)
        return Metric3dModel(cfg=cfg, model=model)

    def get_depth(self, img, fx, fy):
        cv_image = np.array(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        intrinsic = [fx, fy, img.shape[1] / 2, img.shape[0] / 2]
        rgb_input, cam_models_stacks, pad, label_scale_factor = (
            transform_test_data_scalecano(img, intrinsic, self.cfg.data_basic)
        )

        with torch.no_grad():
            pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                model=self.model,
                input=rgb_input,
                cam_model=cam_models_stacks,
                pad_info=pad,
                scale_info=label_scale_factor,
                gt_depth=None,
                normalize_scale=self.cfg.data_basic.depth_range[1],
                ori_shape=[img.shape[0], img.shape[1]],
            )

            pred_normal = output["normal_out_list"][0][:, :3, :, :]
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0] : H - pad[1], pad[2] : W - pad[3]]

        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_depth[pred_depth < 0] = 0

        pred_normal = torch.nn.functional.interpolate(
            pred_normal, [img.shape[0], img.shape[1]], mode="bilinear"
        ).squeeze()
        pred_normal = pred_normal.permute(1, 2, 0)
        pred_normal = pred_normal.cpu().numpy()

        return pred_depth, pred_normal


def _plot3d(xyz, color="b", ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    coords = xyz.reshape(-1, 3)

    ax.scatter(
        coords[:, 0].flatten(),
        coords[:, 1].flatten(),
        coords[:, 2].flatten(),
        s=1,
        c=color,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])


def get_pts_from_depth(
    depth: np.ndarray,
    camera_id: int,
    image_name: str,
    parser: Parser,
):
    cam2world = parser.camtoworlds[camera_id]
    K = parser.Ks_dict[camera_id]
    imsize = parser.imsize_dict[camera_id]
    w2c = np.linalg.inv(cam2world)
    R = w2c[:3, :3]
    C = -R.T @ w2c[:3, 3]
    P = K @ R @ np.hstack([np.eye(3), -C[:, None]])

    sfm_points = parser.points[parser.point_indices[image_name]]

    def get_depth_scalar():
        sfm_points_camera_homo = P @ np.vstack(
            [sfm_points.T, np.ones(sfm_points.shape[0])]
        )
        sfm_points_camera = sfm_points_camera_homo[:2] / sfm_points_camera_homo[2]

        valid_sfm_pt_indices = np.logical_and(
            np.logical_and(sfm_points_camera[0] >= 0, sfm_points_camera[0] < imsize[0]),
            np.logical_and(sfm_points_camera[1] >= 0, sfm_points_camera[1] < imsize[1]),
        )

        if np.sum(valid_sfm_pt_indices) < 10:
            return None, None

        # TODO: beneficial?
        # sfm_point_err = parser.points_err[parser.point_indices[image_name]]
        # valid_sfm_pt_indices = np.logical_and(
        #     valid_sfm_pt_indices, sfm_point_err < 1
        # )

        sfm_points_camera = sfm_points_camera[:, valid_sfm_pt_indices]
        depth_ratios = sfm_points_camera_homo[2, valid_sfm_pt_indices] / (
            1
            + depth[sfm_points_camera[1].astype(int), sfm_points_camera[0].astype(int)]
        )

        depth_scalar = np.mean(depth_ratios)
        # print(f"{depth_scalar=}, {np.std(depth_ratios)=}")
        return depth_scalar, sfm_points_camera_homo

    def transform_camera_to_world_space(camera_homo):
        dense_world = np.linalg.inv(K) @ camera_homo.reshape((-1, 3)).T
        dense_world = (
            cam2world @ np.vstack([dense_world, np.ones(dense_world.shape[1])])
        )[:3].T
        return dense_world.reshape((imsize[1], imsize[0], 3))

    depth_scalar, sfm_points_camera_homo = get_depth_scalar()
    if depth_scalar is None:
        return None

    camera_grid_homo = np.dstack(
        [np.mgrid[0 : imsize[0], 0 : imsize[1]].T, depth_scalar * (1 + depth)]
    )

    camera_grid_homo[:, :, 0] = (camera_grid_homo[:, :, 0] + 0.5) * camera_grid_homo[:, :, 2]
    camera_grid_homo[:, :, 1] = (camera_grid_homo[:, :, 1] + 0.5) * camera_grid_homo[:, :, 2]

    pts = transform_camera_to_world_space(camera_grid_homo)

    camera_plane_xyz = transform_camera_to_world_space(
        np.dstack([np.mgrid[0 : imsize[0], 0 : imsize[1]].T, np.ones(depth.shape)]),
    )[::50, ::50, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # _plot3d(parser.points[::10,], "y", ax)
    _plot3d(sfm_points[::3, :], "y", ax)

    # tmp = sfm_points_camera_homo[0]
    # sfm_points_camera_homo[0] = sfm_points_camera_homo[1]
    # sfm_points_camera_homo[1] = tmp
    sfm_points_camera_homo = (
        np.linalg.inv(K) @ sfm_points_camera_homo.T.reshape((-1, 3)).T
    )
    sfm_points_camera_homo = (
        cam2world
        @ np.vstack([sfm_points_camera_homo, np.ones(sfm_points_camera_homo.shape[1])])
    )[:3].T
    _plot3d(sfm_points_camera_homo, "r", ax)
    _plot3d(pts.reshape(-1, 3)[::30, :], "g", ax)
    _plot3d(camera_plane_xyz, "b", ax)
    plt.show(block=True)

    return pts


def get_init_points_from_metric3d_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    m3d_model = Metric3dModel.load(config, device)

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    downsample_factor = 100

    for i, image_info in enumerate(
        tqdm(
            list(zip(parser.image_paths, parser.image_names)),
            desc="Getting depths with Metric3D",
        )
    ):
        image_path, image_name = image_info
        image = imageio.imread(image_path)
        camera_id: int = parser.camera_ids[i]
        K = parser.Ks_dict[camera_id]
        fx = K[0, 0]
        fy = K[1, 1]
        depth, normal = m3d_model.get_depth(image, fx, fy)
        points = get_pts_from_depth(depth, camera_id, image_name, parser)
        if points is None:
            continue

        if i > 0:
            break

        points = torch.from_numpy(points.reshape([-1, 3])[::downsample_factor, :])
        rgbs = torch.from_numpy(image.reshape([-1, 3])[::downsample_factor, :])
        points_list.append(points)
        rgbs_list.append(rgbs.float() / 255.0)

    return torch.cat(points_list, dim=0).float(), torch.cat(rgbs_list, dim=0).float()
