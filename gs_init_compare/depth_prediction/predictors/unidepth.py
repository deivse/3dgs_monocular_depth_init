import torch

# from unidepth.models import UniDepthV2
from gs_init_compare.config import Config
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)


class UniDepth(DepthPredictor):
    def __init__(self, config: Config, device: str):
        self.model_backbone = config.mdi.unidepth.backbone
        # self.model = UniDepthV2.from_pretrained(
        #     f"lpiccinelli/unidepth-v2-{self.model_backbone}14"
        # ).to(device)

        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v2",
            backbone=f"{self.model_backbone}14",
            pretrained=True,
            trust_repo=True,
            force_reload=False,
        )

        self.device = device

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
        depth = result["depth"].squeeze().to(self.device)
        return PredictedDepth(
            depth=depth,
            mask=torch.ones_like(depth, dtype=torch.bool),
            depth_confidence=result["confidence"].squeeze().to(self.device),
        )
