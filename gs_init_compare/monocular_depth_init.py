import logging
from pathlib import Path
import sys
from typing import List, Type

import numpy as np
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
    InputImage,
    LowDepthAlignmentConfidenceError,
    get_pts_from_depth,
)
from gs_init_compare.point_cloud_postprocess.postprocess import postprocess_point_cloud
from gs_init_compare.utils.cuda_memory import cuda_stats_msg
from gs_init_compare.utils.runner_utils import knn


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
    image_name: str,
    config: Config,
    dataset_name: str,
):
    cache_dir = Path(config.mdi.cache_dir) / model.name / dataset_name

    cache_dir.mkdir(exist_ok=True, parents=True)
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


@torch.no_grad()
def pts_and_rgb_from_monocular_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    _LOGGER.info(cuda_stats_msg(device, "Before loading model"))
    model = pick_model(config)(config, device)
    _LOGGER.info("Using depth predictor model: %s", model.name)

    dataset_name = parser.dataset_name

    _LOGGER.info(cuda_stats_msg(device, "After loading model"))

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    dataset = type(parser).DatasetCls(parser, split="train")
    progress_bar = tqdm(
        dataset,
        desc="Calculating init points from monocular depth",
    )
    intrinsic_matrices = []
    proj_matrices = []
    image_sizes = np.empty((len(dataset), 2), dtype=np.int32)

    _LOGGER.info("Running monocular depth initialization...")
    for i, data in enumerate(progress_bar):
        # Check that the image is actually 0-255
        assert data["image"].max() > 1

        image = InputImage(
            name=data["image_name"],
            cam2world=data["camtoworld"],
            K=data["K"],
            data=data["image"] / 255.0,
        )
        intrinsics = CameraIntrinsics(image.K)

        predicted_depth = predict_depth_or_get_cached_depth(
            model,
            image.data,
            intrinsics,
            image.name,
            config,
            dataset_name,
        )
        assert predicted_depth.depth.device == torch.device(device)

        debug_point_cloud_export_dir = (
            Path(config.mdi.pts_output_dir) / dataset_name / model.name / image.name
            if config.mdi.pts_output_dir and config.mdi.pts_output_per_image
            else None
        )

        try:
            points, subsampling_mask, P = get_pts_from_depth(
                predicted_depth,
                image,
                parser,
                config,
                device,
                debug_point_cloud_export_dir,
            )
        except LowDepthAlignmentConfidenceError as e:
            _LOGGER.warning(
                "Low depth alignment confidence for image %s: {%s}", image.name, e
            )
            continue

        if config.mdi.noise_std_scene_frac is not None:
            points = add_noise_to_point_cloud(
                points, parser.scene_scale * config.mdi.noise_std_scene_frac
            )

        rgbs = image.data.view([-1, 3])[subsampling_mask]

        points_list.append(points)
        rgbs_list.append(rgbs.float())

        intrinsic_matrices.append(image.K.cpu().numpy())
        proj_matrices.append(P.cpu().numpy())
        image_sizes[i] = np.array(image.data.shape[:2][::-1], dtype=np.int32)

        progress_bar.set_description(f"Last processed '{image.name}'", refresh=True)

    pts = torch.cat(points_list, dim=0).float()
    rgbs = torch.cat(rgbs_list, dim=0).float()

    _LOGGER.info("Num points before postprocess: %d", pts.shape[0])
    pts, rgbs = postprocess_point_cloud(
        pts,
        rgbs,
        intrinsic_matrices,
        proj_matrices,
        image_sizes,
        config.mdi.postprocess,
        device,
    )
    _LOGGER.info("Num points after postprocess: %d", pts.shape[0])

    if config.mdi.pts_output_dir is not None:
        output_dir = Path(config.mdi.pts_output_dir) / dataset_name
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{model.name}_{config.mdi.subsample_factor}_{config.mdi.depth_alignment_strategy.value}"
        export_point_cloud_to_ply(
            pts.cpu().numpy(),
            rgbs.cpu().numpy(),
            output_dir,
            filename,
            outlier_std_dev=None,
        )
        export_point_cloud_to_ply(
            parser.points, parser.points_rgb / 255.0, output_dir, "sfm"
        )
        if config.mdi.pts_only:
            sys.exit(0)

    scales = None
    if config.mdi.limit_init_scale:
        dist2_avg = (knn(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        quantile = torch.quantile(dist_avg, config.mdi.init_scale_clamp_quantile)
        dist_avg = torch.clamp(dist_avg, max=quantile)
        scales = (  # [N, 3]
            torch.log(dist_avg * config.init_scale).unsqueeze(-1).repeat(1, 3)
        )
    return pts, rgbs, scales
