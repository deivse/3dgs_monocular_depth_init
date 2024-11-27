import logging

import numpy as np
import torch
from PIL import Image
import torchvision

from third_party.MoGe.moge.model import MoGeModel

from config import Config

from .depth_predictor_interface import DepthPredictor, PredictedDepth, PredictedPoints

_LOGGER = logging.getLogger(__name__)


class MoGe(DepthPredictor):
    def __init__(self, config: Config, device: str):
        # Load the model from huggingface
        self.__device = device
        self.__model = MoGeModel.from_pretrained(
            "Ruicheng/moge-vitl").to(device)

    def can_predict_points_directly(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "MoGe"

    def __infer(self, img: Image.Image) -> torch.Tensor:
        """
        Inference on the input image.

        Args:
            img: The input image.

        Returns: dictionary structured as follows.
            The maps are in the same size as the input image.
            {
                "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
                "depth": (H, W),        # scale-invariant depth map
                "mask": (H, W),         # a binary mask for valid pixels.
                "intrinsics": (3, 3),   # normalized camera intrinsics
            }
        """
        img_tensor = torchvision.transforms.ToTensor()(
            img).to(torch.float32).to(self.__device)
        return self.__model.infer(img_tensor)

    def predict_depth(self, img: Image.Image, *_):
        result = self.__infer(img)
        return PredictedDepth(result["depth"], result["mask"])

    def predict_points(self, img: Image.Image, *_):
        result = self.__infer(img)
        return PredictedPoints(result["points"], result["mask"])
