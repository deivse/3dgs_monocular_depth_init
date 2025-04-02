import logging
from pathlib import Path

import cv2
import torch

from gs_init_compare.config import Config
from gs_init_compare.depth_prediction.utils.download_with_tqdm import (
    download_with_pbar,
)

from gs_init_compare.third_party.depth_anything_v2.metric_depth.depth_anything_v2.dpt import (
    DepthAnythingV2 as DepthAnythingV2Model,
)

from .depth_predictor_interface import DepthPredictor, PredictedDepth

_LOGGER = logging.getLogger(__name__)


class DepthAnythingV2(DepthPredictor):
    __MODEL_CONFIGS = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    _MODEL_PARAMS_BY_TYPE = {
        "indoor": {
            "dataset": "hypersim",
            "max_depth": 20,
        },
        "outdoor": {
            "dataset": "vkitti",
            "max_depth": 80,
        },
    }

    def __init__(self, config: Config, device: str):
        self.device = device
        self.encoder = config.mdi.depthanything.backbone
        self.model_type = config.mdi.depthanything.model_type

        try:
            dataset = self._MODEL_PARAMS_BY_TYPE[self.model_type]["dataset"]
            max_depth = self._MODEL_PARAMS_BY_TYPE[self.model_type]["max_depth"]
        except KeyError as e:
            raise ValueError(
                f"Unsupported model type '{self.model_type}' for DepthAnythingV2"
            ) from e

        checkpoint_path = (
            Path(config.mdi.cache_dir)
            / f"checkpoints/depth_anything_v2_metric_{self.encoder}_{dataset}.pt"
        )

        if not checkpoint_path.exists():
            # Download the checkpoint if it doesn't exist
            url = self.__get_checkpoint_url(self.encoder, dataset)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                f"Downloading DepthAnythingV2 Metric {self.model_type.capitalize()} ({self.encoder}) checkpoint from {url} to {str(checkpoint_path)}"
            )
            download_with_pbar(url, checkpoint_path)

        self.model = DepthAnythingV2Model(
            **{**self.__MODEL_CONFIGS[self.encoder], "max_depth": max_depth}
        )
        self.model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
        self.model = self.model.to(device).eval()

    @staticmethod
    def __get_checkpoint_url(encoder: str, dataset: str) -> str:
        encoder_to_name = {
            "vits": "Small",
            "vitb": "Base",
            "vitl": "Large",
            # "vitg": "Giant", # Not available yet
        }

        dataset_to_name = {
            "hypersim": "Hypersim",
            "vkitti": "VKITTI",
        }

        return (
            "https://huggingface.co/depth-anything/"
            f"Depth-Anything-V2-Metric-{dataset_to_name[dataset]}-{encoder_to_name[encoder]}/"
            f"resolve/main/depth_anything_v2_metric_{dataset}_{encoder}.pth?download=true"
        )

    def can_predict_points_directly(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"DepthAnythingV2_{self.encoder}_{self.model_type}"

    def predict_depth(self, img: torch.Tensor, *_) -> PredictedDepth:
        # `infer_image` expects image in BGR and in range [0, 255]
        input_image = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2RGB) * 255
        depth = self.model.infer_image(input_image)
        return PredictedDepth(torch.from_numpy(depth).to(self.device), None)
