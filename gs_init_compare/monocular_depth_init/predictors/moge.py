import logging

import numpy as np
import torch
from PIL import Image
import torchvision

from gs_init_compare.third_party.MoGe.moge.model import MoGeModel

from gs_init_compare.config import Config

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

    def __preprocess(self, img: torch.Tensor):
        assert img.ndim == 3
        return img.permute(2, 1, 0).to(self.__device)

    def predict_depth(self, img: torch.Tensor, *_):
        result = self.__model.infer(self.__preprocess(img))
        return PredictedDepth(result["depth"], result["mask"])

    def predict_points(self, img: torch.Tensor, *_):
        result = self.__model.infer(self.__preprocess(img))
        return PredictedPoints(result["points"], result["mask"])
