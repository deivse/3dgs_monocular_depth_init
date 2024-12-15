import logging
import urllib.request
from copy import deepcopy
from pathlib import Path

import depth_pro
import numpy as np
from PIL import Image

from gs_init_compare.config import Config
from gs_init_compare.monocular_depth_init.utils.download_with_tqdm import download_with_pbar

from .depth_predictor_interface import DepthPredictor, PredictedDepth

_LOGGER = logging.getLogger(__name__)


def _load_rgb(pil_img: Image.Image, auto_rotate: bool, remove_alpha: bool):
    """
    A "fork" of `depth_pro.utils.load_rgb` that accepts an already loaded PIL image.
    """

    img_exif = depth_pro.utils.extract_exif(pil_img)
    icc_profile = pil_img.info.get("icc_profile", None)

    # Rotate the image.
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            pil_img = pil_img.transpose(Image.Transpose.ROTATE_180)
        elif exif_orientation == 6:
            pil_img = pil_img.transpose(Image.Transpose.ROTATE_270)
        elif exif_orientation == 8:
            pil_img = pil_img.transpose(Image.Transpose.ROTATE_90)
        elif exif_orientation != 1:
            _LOGGER.warning(f"Ignoring image orientation {exif_orientation}.")

    img = np.array(pil_img)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    _LOGGER.debug(f"\tHxW: {img.shape[0]}x{img.shape[1]}")

    # Extract the focal length from exif data.
    f_35mm = img_exif.get(
        "FocalLengthIn35mmFilm",
        img_exif.get(
            "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
        ),
    )
    if f_35mm is not None and f_35mm > 0:
        _LOGGER.debug(f"\tfocal length @ 35mm film: {f_35mm}mm")
        f_px = depth_pro.utils.fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    else:
        f_px = None

    return img, icc_profile, f_px


class AppleDepthPro(DepthPredictor):
    def __init__(self, config: Config, device: str):
        # Load model and preprocessing transform
        depth_pro_config = deepcopy(
            depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT)
        depth_pro_config.checkpoint_uri = config.depth_pro_checkpoint

        checkpoint_path = Path(config.depth_pro_checkpoint)
        if not checkpoint_path.exists():
            # Download the checkpoint if it doesn't exist
            url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                f"Downloading DepthPro checkpoint from {url} to {str(checkpoint_path)}")
            download_with_pbar(url, checkpoint_path)

        self.__model, self.__transform = depth_pro.create_model_and_transforms(
            depth_pro_config, device
        )

        self.__model.eval()

    def can_predict_points_directly(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "AppleDepthPro"

    def predict_depth(self, img: Image.Image, *_) -> PredictedDepth:
        raise NotImplementedError(
            "AppleDepthPro needs exif data that is not available via the Dataset right now."
        )
        # Load and preprocess an image.
        image, _, f_px = _load_rgb(img, auto_rotate=True, remove_alpha=True)
        image = self.__transform(img)

        # Run inference.
        prediction = self.__model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        return PredictedDepth(depth, None)
