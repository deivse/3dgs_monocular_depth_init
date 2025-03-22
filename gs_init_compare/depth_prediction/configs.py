from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DepthAnythingV2Config:
    """
    Configuration for the Depth Anything V2 monocular depth predictor.
    """

    # Select backbone for Depth Anything V2 model
    backbone: Literal["vits", "vitb", "vitl"] = "vitl"
    # Select model type for Depth Anything
    model_type: Literal["indoor", "outdoor"] = "indoor"


@dataclass
class Metric3dV2Config:
    """
    Configuration for the Metric3dV2 monocular depth predictor.
    """

    # Path to Metric3d config file. Must be set if using Metric3D as the depth predictor.
    config: Optional[str] = None
    # Path to Metric3d checkpoint. Must be set if using Metric3D as the depth predictor.
    weights: Optional[str] = None


@dataclass
class UnidepthConfig:
    """
    Configuration for the UniDepth monocular depth predictor.
    """

    # Select backbone for UniDepth model if using "unidepth" mono_depth_model.
    backbone: Literal["vitl14", "cnvnxtl"] = "vitl14"
