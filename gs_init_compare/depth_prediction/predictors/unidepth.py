import torch

from unidepth.models import UniDepthV1
from gs_init_compare.config import Config
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
    PredictedPoints,
)


class UniDepth(DepthPredictor):
    def __init__(self, config: Config, device: str):
        self.model_backbone = config.mdi.unidepth.backbone
        self.model = UniDepthV1.from_pretrained(
            f"lpiccinelli/unidepth-v1-{self.model_backbone}"
        ).to(device)
        self.device = device

    def can_predict_points_directly(self) -> bool:
        # Return whether the model can predict points directly
        return True

    def __preprocess(self, img: torch.Tensor):
        assert img.ndim == 3
        return img.permute(2, 0, 1).to(self.device)

    @property
    def name(self) -> str:
        # Return the name of the predictor
        return f"UniDepth_{self.model_backbone}"

    def __predict(self, img: torch.Tensor, intrinsics: CameraIntrinsics):
        rgb = self.__preprocess(img)
        return self.model.infer(rgb, intrinsics.K)

    def predict_depth(self, img: torch.Tensor, intrinsics: CameraIntrinsics):
        result = self.__predict(img, intrinsics)
        return PredictedDepth(result["depth"].squeeze(), None)

    def predict_points(self, img: torch.Tensor, intrinsics: CameraIntrinsics):
        result = self.__predict(img, intrinsics)
        return PredictedPoints(result["points"].squeeze(), None)
