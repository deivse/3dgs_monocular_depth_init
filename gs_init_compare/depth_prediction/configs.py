from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass
class DepthAnythingV2Config:
    """
    Configuration for the Depth Anything V2 monocular depth predictor.
    """

    # Select backbone for Depth Anything V2 model
    backbone: Literal["vits", "vitb", "vitl"] = "vitl"
    # Select model type for Depth Anything
    model_type: Literal["indoor", "outdoor"] = "indoor"


class Metric3dPreset(str, Enum):
    vit_large = "vit_large"
    vit_small = "vit_small"


@dataclass
class Metric3dV2Config:
    """
    Configuration for the Metric3dV2 monocular depth predictor.
    """

    preset: Metric3dPreset = Metric3dPreset.vit_large


@dataclass
class UnidepthConfig:
    """
    Configuration for the UniDepth monocular depth predictor.
    """

    # Select backbone for UniDepth model if using "unidepth" mono_depth_model.
    backbone: Literal["vitl14", "cnvnxtl"] = "vitl14"
