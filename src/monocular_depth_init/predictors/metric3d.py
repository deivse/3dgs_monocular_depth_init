import logging

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from third_party.metric3d.mono.model.monodepth_model import (
    get_configured_monodepth_model,
)
from third_party.metric3d.mono.utils.do_test import (
    get_prediction,
    transform_test_data_scalecano,
)
from third_party.metric3d.mono.utils.running import load_ckpt
from monocular_depth_init.predictors.depth_predictor_interface import DepthPredictor

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
        self.cfg = Metric3dConfig.fromfile(config.metric3d_config)
        self.model, _, _, _ = load_ckpt(
            config.metric3d_weights,
            get_configured_monodepth_model(
                self.cfg,
            ),
            strict_match=False,
        )
        self.model.eval()
        self.model.to(device)

    @property
    def name(self) -> str:
        return self.__name

    def can_predict_points_directly(self) -> bool:
        return False

    def predict_depth(self, img: Image.Image, fx: float, fy: float) -> torch.Tensor:
        cv_image = np.asarray(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        intrinsic = [fx, fy, img.shape[1] / 2, img.shape[0] / 2]
        rgb_input, cam_models_stacks, pad, label_scale_factor = (
            transform_test_data_scalecano(img, intrinsic, self.cfg.data_basic)
        )

        with torch.no_grad():
            pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                model=self.model,
                input=rgb_input,
                cam_model=cam_models_stacks,
                pad_info=pad,
                scale_info=label_scale_factor,
                gt_depth=None,
                normalize_scale=self.cfg.data_basic.depth_range[1],
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

        return pred_depth
