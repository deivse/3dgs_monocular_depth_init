from dataclasses import dataclass
import pickle
import imageio
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


def get_pts_from_depth(
    depth: np.ndarray,
    camera_id: int,
    image_name: str,
    parser: Parser,
):
    # Save function inputs as pickle
    # with open("rick.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "depth": depth,
    #             "camera_id": camera_id,
    #             "image_name": image_name,
    #             "parser": parser,
    #         },
    #         f,
    #     )

    cam2world = parser.camtoworlds[camera_id]
    K = parser.Ks_dict[camera_id]
    imsize = parser.imsize_dict[camera_id]

    sfm_points = parser.points[parser.point_indices[image_name]]
    world2cam = np.linalg.inv(cam2world)
    P = K @ world2cam[:3]

    sfm_points_camera = P @ np.vstack([sfm_points.T, np.ones(sfm_points.shape[0])])
    sfm_points_camera = sfm_points_camera[:2] / sfm_points_camera[2]
    valid_sfm_pt_indices = np.logical_and(
        np.logical_and(sfm_points_camera[0] >= 0, sfm_points_camera[0] < imsize[0]),
        np.logical_and(sfm_points_camera[1] >= 0, sfm_points_camera[1] < imsize[1]),
    )
    valid_sfm_pt_indices = np.logical_and(valid_sfm_pt_indices, sfm_points[:, 2] > 0)
    sfm_points_camera = sfm_points_camera[:, valid_sfm_pt_indices]

    depth_ratios = (
        sfm_points[valid_sfm_pt_indices, 2]
        / depth[sfm_points_camera[1].astype(int), sfm_points_camera[0].astype(int)]
    )
    depth_scalar = np.mean(depth_ratios)
    print(f"{depth_scalar=}, {np.std(depth_ratios)=}")

    camera_grid = np.dstack(
        [np.mgrid[0 : imsize[1], 0 : imsize[0]].T, depth.T * depth_scalar]
    )
    xyz = (np.linalg.inv(K) @ camera_grid.reshape(-1, 3).T).T
    xyz = xyz.reshape(imsize[0], imsize[1], 3)

    torch.nn.functional.grid_sample(
        torch.tensor(xyz).permute(2, 0, 1).unsqueeze(0).float(),
        torch.tensor(sfm_points).unsqueeze(0).float(),
    )


def get_init_points_from_metric3d_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    m3d_model = Metric3dModel.load(config, device)

    points: torch.Tensor
    rgbs: torch.Tensor

    points_list = []
    rgbs_list = []

    for i, image_info in enumerate(
        tqdm(
            zip(parser.image_paths, parser.image_names),
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

    return torch.tensor(
        [[0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 2, 2], [0, 3, 1]]
    ).float(), torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]
    ).float()
