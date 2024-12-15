import logging
from pathlib import Path
import sys
from typing import List, Type

from PIL import Image
import numpy as np
import torch
import open3d
from tqdm import tqdm

from gs_init_compare.config import Config
from gs_init_compare.datasets.colmap import Parser
from gs_init_compare.monocular_depth_init.predictors.depth_predictor_interface import (
    DepthPredictor,
)
from gs_init_compare.monocular_depth_init.utils.points_from_depth import (
    DebugPlotConfig,
    LowDepthAlignmentConfidenceError,
    get_pts_from_depth,
)
from gs_init_compare.utils.cuda_memory import cuda_stats_msg


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
        raise ValueError(f"Unsupported monodepth model: {config.mono_depth_model}")


def save_pts_and_rgbs(
    pts: np.ndarray, rgbs: np.ndarray, output_dir: Path, depth_pts_filename: str
):
    """Saves point cloud to a .ply file."""
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    pcd.colors = open3d.utility.Vector3dVector(rgbs)
    open3d.io.write_point_cloud(
        str(output_dir / f"{depth_pts_filename}.ply"), pcd, write_ascii=False
    )
    logging.info(f"Saved point cloud to {output_dir / depth_pts_filename}.ply")


def predict_depth_or_get_cached_depth(
    model: DepthPredictor,
    image: torch.Tensor,
    fx: float,
    fy: float,
    image_id,
    config: Config,
    dataset_name: str,
):
    cache_dir = Path(config.mono_depth_cache_dir) / model.name / dataset_name

    cache_dir.mkdir(exist_ok=True, parents=True)
    image_name = str(image_id)
    cache_path = cache_dir / f"{image_name}.pth"

    depth = None
    if not config.ignore_mono_depth_cache and cache_path.exists():
        try:
            depth = torch.load(cache_path, weights_only=True)
        except Exception as e:
            _LOGGER.warning(f"Failed to load cached depth for image {image_name}: {e}")

    # TODO: support for models that can predict points directly
    if depth is None:
        depth = model.predict_depth(image, fx, fy)
        try:
            torch.save(depth, cache_path)
        except KeyboardInterrupt:
            cache_path.unlink(missing_ok=True)
            raise
    return depth


def pts_and_rgb_from_monocular_depth(
    config: Config, parser: Parser, device: str = "cuda"
):
    print(cuda_stats_msg(device, "Before loading model"))
    model = pick_model(config)(config, device)
    dataset_name = parser.dataset_name

    print(cuda_stats_msg(device, "After loading model"))

    points_list: List[torch.Tensor] = []
    rgbs_list: List[torch.Tensor] = []

    downsample_factor = config.dense_depth_downsample_factor
    dataset = type(parser).DatasetCls(parser, split="train")
    progress_bar = tqdm(
        dataset,
        desc="Calculating init points from monocular depth",
    )
    print("Running monocular depth initialization...")
    for data in progress_bar:
        image_id = data["image_id"]
        cam2world = data["camtoworld"]
        image_name = parser.image_names[image_id]
        K = data["K"]
        fx = K[0, 0]
        fy = K[1, 1]

        # Check that the image is actually 0-255
        assert data["image"].max() > 1
        image: torch.Tensor = data["image"] / 255.0

        with torch.no_grad():
            depth, mask = predict_depth_or_get_cached_depth(
                model, image, fx, fy, image_id, config, dataset_name
            )
            cpu_depth = depth.cpu().numpy()
            if mask is not None:
                cpu_depth[mask.cpu().numpy() == 0] = 0

        try:
            points, valid_point_indices, inlier_ratio = get_pts_from_depth(
                depth,
                mask,
                image_id,
                parser,
                cam2world,
                K,
                downsample_factor=downsample_factor,
                debug_plot_conf=None,
                # debug_plot_conf=DebugPlotConfig(),
            )
        except LowDepthAlignmentConfidenceError as e:
            _LOGGER.warning(
                f"Low depth alignment confidence for image {image_name}: {e}"
            )
            continue
        progress_bar.set_description(
            f"Last processed '{image_name}',"
            f" (inlier depth ratio {inlier_ratio:.2f})",
            refresh=True,
        )

        if points is None:
            _LOGGER.warning(f"Failed to get points for image {image_name}")
            continue

        rgbs = image[::downsample_factor, ::downsample_factor, :].reshape([-1, 3])
        # inlier indices are for a downsampled and flattened array
        rgbs = rgbs[valid_point_indices]
        points_list.append(points)
        rgbs_list.append(rgbs.float())
    print(cuda_stats_msg(device, "After processing points"))

    pts = torch.cat(points_list, dim=0).float()
    rgbs = torch.cat(rgbs_list, dim=0).float()

    if config.mono_depth_pts_output_dir is not None:
        output_dir = Path(config.mono_depth_pts_output_dir) / dataset_name
        output_dir.mkdir(exist_ok=True, parents=True)
        save_pts_and_rgbs(pts.cpu().numpy(), rgbs.cpu().numpy(), output_dir, model.name)
        save_pts_and_rgbs(parser.points, parser.points_rgb / 255.0, output_dir, "sfm")
        if config.mono_depth_pts_only:
            sys.exit(0)

    return pts, rgbs
