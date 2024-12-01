import logging
from pathlib import Path
import shutil
from typing import List, Type

from PIL import Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from config import Config
from datasets.colmap import Parser
from monocular_depth_init.predictors.depth_predictor_interface import DepthPredictor
from monocular_depth_init.utils.points_from_depth import (
    DebugPlotConfig,
    get_pts_from_depth,
)


_LOGGER = logging.getLogger(__name__)


def pick_model(config: Config) -> Type[DepthPredictor]:
    _LOGGER.info(f"Using depth predictor model: {config.mono_depth_model}")
    if config.mono_depth_model is None:
        raise ValueError("No depth predictor model specified in config.")

    if config.mono_depth_model == "metric3d":
        from .predictors.metric3d import Metric3d

        return Metric3d
    elif config.mono_depth_model == "depth_pro":
        from .predictors.apple_depth_pro import AppleDepthPro

        return AppleDepthPro
    elif config.mono_depth_model == "moge":
        from .predictors.moge import MoGe

        return MoGe
    else:
        raise ValueError(
            f"Unsupported monodepth model: {config.mono_depth_model}")


def predict_depth_or_get_cached_depth(
    model: DepthPredictor,
    pil_image: Image.Image,
    fx: float,
    fy: float,
    image_name: str,
    config: Config,
):
    dataset_name = config.data_dir.removeprefix(
        "data/360_v2").replace("/", "_")
    cache_dir = Path(config.mono_depth_cache_dir) / model.name / dataset_name
    if config.invalidate_mono_depth_cache:
        _LOGGER.info(
            "Invalidating monocular depth cache dir %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / f"{image_name}.pth"

    depth = None
    if cache_path.exists():
        try:
            depth = torch.load(cache_path, weights_only=True)
        except Exception as e:
            _LOGGER.warning(
                f"Failed to load cached depth for image {image_name}: {e}")

    # TODO: support for models that can predict points directly
    if depth is None:
        depth = model.predict_depth(pil_image, fx, fy)
        try:
            torch.save(depth, cache_path)
        except KeyboardInterrupt:
            cache_path.unlink(missing_ok=True)
            raise
    return depth


def pts_and_rgb_from_monocular_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    model = pick_model(config)(config, device)

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    downsample_factor = config.dense_depth_downsample_factor
    progress_bar = tqdm(
        list(zip(parser.image_paths, parser.image_names)),
        desc="Calculating init points from monocular depth",
    )
    print("Running monocular depth initialization...")
    for i, image_info in enumerate(progress_bar):
        image_path, image_name = image_info
        pil_image = Image.open(image_path)
        pil_image.load()
        camera_id: int = parser.camera_ids[i]
        K = parser.Ks_dict[camera_id]
        fx = K[0, 0]
        fy = K[1, 1]

        if False:  # model.can_predict_points_directly():
            points, valid_point_indices = model.predict_points(
                pil_image, fx, fy
            )
        else:
            depth, mask = predict_depth_or_get_cached_depth(
                model, pil_image, fx, fy, image_name, config
            )
            points, valid_point_indices, inlier_ratio = get_pts_from_depth(
                depth,
                mask,
                image_name,
                i,
                parser,
                downsample_factor=downsample_factor,
                debug_plot_conf=None,
                # debug_plot_conf=DebugPlotConfig(),
            )
            progress_bar.set_description(
                f"Last processed '{image_name}',"
                f" (inlier depth ratio {inlier_ratio:.2f})", refresh=True)

        if points is None:
            _LOGGER.warning(f"Failed to get points for image {image_name}")
            continue

        image = np.asarray(pil_image)
        rgbs = image[::downsample_factor,
                     ::downsample_factor, :].reshape([-1, 3])
        # inlier indices are for a downsampled and flattened array
        rgbs = torch.from_numpy(rgbs[valid_point_indices])
        points_list.append(points)
        rgbs_list.append(rgbs.float() / 255.0)

    return torch.cat(points_list, dim=0).float(), torch.cat(rgbs_list, dim=0).float()
