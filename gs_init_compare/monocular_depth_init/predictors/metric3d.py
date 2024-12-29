import logging

import cv2
import numpy as np
import torch
from PIL import Image

from gs_init_compare.config import Config
from gs_init_compare.third_party.metric3d.mono.model.monodepth_model import (
    get_configured_monodepth_model,
)
from gs_init_compare.third_party.metric3d.mono.utils.do_test import (
    get_prediction,
    transform_test_data_scalecano,
)
from gs_init_compare.third_party.metric3d.mono.utils.running import load_ckpt
from gs_init_compare.monocular_depth_init.predictors.depth_predictor_interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)

try:
    from mmcv.utils import Config as Metric3dConfig
except ImportError:
    from mmengine import Config as Metric3dConfig

_LOGGER = logging.getLogger(__name__)


class Metric3d(DepthPredictor):
    def __init__(self, config: Config, device: str):
        if config.metric3d_config is None:
            raise ValueError("Metric3d config path is not provided.")
        if config.metric3d_weights is None:
            raise ValueError("Metric3d weights path is not provided.")

        self.__name = f"Metric3d_{config.metric3d_config.split('.')[-2]}"
        self.__cfg = Metric3dConfig.fromfile(config.metric3d_config)
        self.__model, _, _, _ = load_ckpt(
            config.metric3d_weights,
            get_configured_monodepth_model(
                self.__cfg,
            ),
            strict_match=False,
        )
        self.__model.eval()
        self.__model.to(device)

    @property
    def name(self) -> str:
        return self.__name

    def can_predict_points_directly(self) -> bool:
        return False

    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        img = img.cpu().numpy() * 255.0
        intrinsic = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
        rgb_input, cam_models_stacks, pad, label_scale_factor = (
            transform_test_data_scalecano(img, intrinsic, self.__cfg.data_basic)
        )

        with torch.no_grad():
            pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                model=self.__model,
                input=rgb_input,
                cam_model=cam_models_stacks,
                pad_info=pad,
                scale_info=label_scale_factor,
                gt_depth=None,
                normalize_scale=self.__cfg.data_basic.depth_range[1],
                ori_shape=[img.shape[0], img.shape[1]],
            )

            pred_normal = output["normal_out_list"][0][:, :3, :, :]
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0] : H - pad[1], pad[2] : W - pad[3]]

        pred_depth = pred_depth.squeeze()
        pred_depth[pred_depth < 0] = 0

        # pred_normal = torch.nn.functional.interpolate(
        #     pred_normal, [img.shape[0], img.shape[1]], mode="bilinear"
        # ).squeeze()
        # pred_normal = pred_normal.permute(1, 2, 0)
        # pred_normal = pred_normal.cpu().numpy()

        return PredictedDepth(pred_depth, None)
