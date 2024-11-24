import logging
from typing import List, Type

import imageio
import torch
from tqdm import tqdm

from config import Config
from datasets.colmap import Parser
from monocular_depth_init.predictors.depth_predictor_interface import DepthPredictor
from monocular_depth_init.utils.points_from_depth import get_pts_from_depth

from .predictors.metric3d import Metric3d

_LOGGER = logging.getLogger(__name__)


def pick_model(config: Config) -> Type[DepthPredictor]:
    _LOGGER.info(f"Using depth predictor model: {config.mono_depth_model}")
    if config.mono_depth_model is None:
        raise ValueError("No depth predictor model specified in config.")

    if config.mono_depth_model == "metric3d":
        return Metric3d
    else:
        raise ValueError(f"Unsupported monodepth model: {config.mono_depth_model}")


def pts_and_rgb_from_monocular_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    model = pick_model(config)(config, device)

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    downsample_factor = config.dense_depth_downsample_factor
    for i, image_info in enumerate(
        tqdm(
            list(zip(parser.image_paths, parser.image_names)),
            desc="Calculating init points from monocular depth",
        )
    ):
        image_path, image_name = image_info
        image = imageio.imread(image_path)
        camera_id: int = parser.camera_ids[i]
        K = parser.Ks_dict[camera_id]
        fx = K[0, 0]
        fy = K[1, 1]

        if model.can_predict_points_directly():
            points, valid_point_indices = model.predict_3d_point_cloud(image, fx, fy)
        else:
            depth = model.predict_depth(image, fx, fy)
            points, valid_point_indices = get_pts_from_depth(
                depth, image_name, i, parser, downsample_factor=downsample_factor
            )

        if points is None:
            _LOGGER.warning(f"Failed to get points for image {image_name}")
            continue

        rgbs = image[::downsample_factor, ::downsample_factor, :].reshape([-1, 3])
        # inlier indices are for a downsampled and flattened array
        rgbs = torch.from_numpy(rgbs[valid_point_indices])
        points_list.append(points)
        rgbs_list.append(rgbs.float() / 255.0)

    return torch.cat(points_list, dim=0).float(), torch.cat(rgbs_list, dim=0).float()
