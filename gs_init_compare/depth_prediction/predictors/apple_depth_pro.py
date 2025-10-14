import logging
from copy import deepcopy
from pathlib import Path

import depth_pro
import numpy as np
from PIL import Image
import torch

from gs_init_compare.config import Config
from gs_init_compare.utils.download_with_tqdm import (
    download_with_pbar,
)

from .depth_predictor_interface import CameraIntrinsics, DepthPredictor, PredictedDepth

_LOGGER = logging.getLogger(__name__)


def _load_rgb(img: torch.Tensor, intrinsics: CameraIntrinsics):
    icc_profile = None
    _LOGGER.debug(f"abs(fx - fy) = {abs(intrinsics.fx - intrinsics.fy)}")
    # Should be equal, but may be slightly inconsistent
    f_px = torch.tensor([intrinsics.fx + intrinsics.fy]) / 2.0

    return img.cpu().numpy(), icc_profile, f_px


class AppleDepthPro(DepthPredictor):
    def __init__(self, config: Config, device: str):
        checkpoint_path = Path(config.mdi.cache_dir) / "checkpoints/depth_pro.pt"

        depth_pro_config = deepcopy(depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT)
        depth_pro_config.checkpoint_uri = str(checkpoint_path)

        # Download the checkpoint if it doesn't exist
        url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_with_pbar(url, checkpoint_path)

        # Load model and preprocessing transform
        self.__model, self.__transform = depth_pro.create_model_and_transforms(
            depth_pro_config, device
        )

        self.__model.eval().to(device)
        self.device = device

    def can_predict_points_directly(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "AppleDepthPro"

    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        # Load and preprocess an image.
        img, _, f_px = _load_rgb(img, intrinsics)
        img = self.__transform(img)

        # Run inference.
        prediction = self.__model.infer(img, f_px=f_px.to(self.device))
        depth = prediction["depth"]  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        return PredictedDepth(depth, torch.ones_like(depth, dtype=torch.bool))
