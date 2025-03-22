import logging
from pathlib import Path
import sys
from typing import List, Type

import torch
from tqdm import tqdm

from gs_init_compare.config import Config
from gs_init_compare.datasets.colmap import Parser
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    CameraIntrinsics,
    DepthPredictor,
)
from gs_init_compare.depth_prediction.utils.point_cloud_export import (
    export_point_cloud_to_ply,
)
from gs_init_compare.depth_prediction.points_from_depth import (
    LowDepthAlignmentConfidenceError,
    get_pts_from_depth,
)
from gs_init_compare.utils.cuda_memory import cuda_stats_msg


_LOGGER = logging.getLogger(__name__)


def pick_model(config: Config) -> Type[DepthPredictor]:
    if config.mdi.predictor is None:
        raise ValueError("No depth predictor model specified in config.")

    if config.mdi.predictor == "metric3d":
        from .depth_prediction.predictors.metric3d import Metric3d

        return Metric3d
    elif config.mdi.predictor == "depth_pro":
        from .depth_prediction.predictors.apple_depth_pro import AppleDepthPro

        return AppleDepthPro
    elif config.mdi.predictor == "moge":
        from .depth_prediction.predictors.moge import MoGe

        return MoGe
    elif config.mdi.predictor == "unidepth":
        from .depth_prediction.predictors.unidepth import UniDepth

        return UniDepth
    elif config.mdi.predictor == "depth_anything_v2":
        from .depth_prediction.predictors.depth_anything_v2 import DepthAnythingV2

        return DepthAnythingV2
    else:
        raise ValueError(f"Unsupported monodepth model: {config.mdi.predictor}")


def predict_depth_or_get_cached_depth(
    model: DepthPredictor,
    image: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image_id,
    config: Config,
    dataset_name: str,
):
    cache_dir = Path(config.mdi.cache_dir) / model.name / dataset_name

    cache_dir.mkdir(exist_ok=True, parents=True)
    image_name = str(image_id)
    cache_path = cache_dir / f"{image_name}.pth"

    depth = None
    if not config.mdi.ignore_cache and cache_path.exists():
        try:
            depth = torch.load(cache_path)
        except Exception as e:
            _LOGGER.warning(f"Failed to load cached depth for image {image_name}: {e}")

    # TODO: support for models that can predict points directly
    if depth is None:
        depth = model.predict_depth(image, intrinsics)
        try:
            torch.save(depth, cache_path)
        except KeyboardInterrupt:
            cache_path.unlink(missing_ok=True)
            raise
    return depth


def add_noise_to_point_cloud(pts: torch.Tensor, noise_std: float):
    noise = torch.randn_like(pts) * noise_std
    return pts + noise


def pts_and_rgb_from_monocular_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    print(cuda_stats_msg(device, "Before loading model"))
    model = pick_model(config)(config, device)
    _LOGGER.info(f"Using depth predictor model: {model.name}")

    dataset_name = parser.dataset_name

    print(cuda_stats_msg(device, "After loading model"))

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    downsample_factor = config.mdi.subsample_factor
    dataset = type(parser).DatasetCls(parser, split="train")
    progress_bar = tqdm(
        dataset,
        desc="Calculating init points from monocular depth",
    )
    print("Running monocular depth initialization...")
    for data in progress_bar:
        image_id = data["image_id"]
        cam2world = data["camtoworld"]
        image_name = data["image_name"]
        K = data["K"]
        intrinsics = CameraIntrinsics(K)

        # Check that the image is actually 0-255
        assert data["image"].max() > 1
        image: torch.Tensor = data["image"] / 255.0

        with torch.no_grad():
            predicted_depth = predict_depth_or_get_cached_depth(
                model, image, intrinsics, image_id, config, dataset_name
            )

        try:
            # TODO: Adaptive downsampling as optional feature!!
            points, adaptive_ds_mask, valid_point_indices = get_pts_from_depth(
                predicted_depth,
                image,
                image_name,
                parser,
                cam2world,
                K,
                config.mdi.depth_alignment_strategy,
                # downsample_factor=downsample_factor,
                debug_point_cloud_export_dir=(
                    Path(config.mdi.pts_output_dir)
                    / dataset_name
                    / model.name
                    / image_name
                    if config.mdi.pts_output_dir and config.mdi.pts_output_per_image
                    else None
                ),
            )

            if config.mdi.noise_std_scene_frac is not None:
                points = add_noise_to_point_cloud(
                    points, parser.scene_scale * config.mdi.noise_std_scene_frac
                )

        except LowDepthAlignmentConfidenceError as e:
            _LOGGER.warning(
                f"Low depth alignment confidence for image {image_name}: {e}"
            )
            continue
        progress_bar.set_description(
            f"Last processed '{image_name}'",
            refresh=True,
        )

        if points is None:
            _LOGGER.warning(f"Failed to get points for image {image_name}")
            continue

        rgbs = image.view([-1, 3])[adaptive_ds_mask]
        # valid point indices are for a downsampled and flattened array
        rgbs = rgbs[valid_point_indices]
        points_list.append(points)
        rgbs_list.append(rgbs.float())

    pts = torch.cat(points_list, dim=0).float()
    rgbs = torch.cat(rgbs_list, dim=0).float()

    if config.mono_depth_pts_output_dir is not None:
        output_dir = Path(config.mono_depth_pts_output_dir) / dataset_name
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{model.name}_{config.mdi.depth_alignment_strategy.value}"
        export_point_cloud_to_ply(
            pts.cpu().numpy(),
            rgbs.cpu().numpy(),
            output_dir,
            filename,
            outlier_std_dev=5,
        )
        export_point_cloud_to_ply(
            parser.points, parser.points_rgb / 255.0, output_dir, "sfm"
        )
        if config.mono_depth_pts_only:
            sys.exit(0)

    return pts, rgbs
