import logging
from pathlib import Path
from typing import Tuple

import torch

from gs_init_compare.config import Config
from gs_init_compare.depth_prediction.configs import Metric3dPreset
from gs_init_compare.utils.download_with_tqdm import download_with_pbar
from gs_init_compare.third_party.metric3d.mono.model.monodepth_model import (
    get_configured_monodepth_model,
)
from gs_init_compare.third_party.metric3d.mono.utils.do_test import (
    get_prediction,
    transform_test_data_scalecano,
)
from gs_init_compare.third_party.metric3d.mono.utils.running import load_ckpt
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)

try:
    from mmcv.utils import Config as Metric3dConfig
except ImportError:
    from mmengine import Config as Metric3dConfig

_LOGGER = logging.getLogger(__name__)


def weights_url_by_name(name):
    return f"https://huggingface.co/spaces/JUGGHM/Metric3D/resolve/main/weight/{name}?download=true"


def get_config_and_weights_name_by_preset(config: Config) -> Tuple[Path, str]:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
    CONFIG_FILENAMES = {
        Metric3dPreset.vit_large: "vit.raft5.large.py",
        Metric3dPreset.vit_small: "vit.raft5.small.py",
    }
    WEIGHTS_FILENAMES = {
        Metric3dPreset.vit_large: "metric_depth_vit_large_800k.pth",
        Metric3dPreset.vit_small: "metric_depth_vit_small_800k.pth",
    }
    preset = config.mdi.metric3d.preset
    config = (
        PROJECT_ROOT
        / "third_party/metric3d/mono/configs/HourglassDecoder"
        / CONFIG_FILENAMES[preset]
    )

    return config, WEIGHTS_FILENAMES[preset]


class Metric3d(DepthPredictor):
    def __init__(self, config: Config, device: str):
        config_path, weights_filename = get_config_and_weights_name_by_preset(config)

        weights_path = Path(config.mdi.cache_dir) / "checkpoints" / weights_filename

        if not weights_path.exists():
            # Download the checkpoint if it doesn't exist
            url = weights_url_by_name(weights_filename)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                f"Downloading Metric3dV2 checkpoint from {url} to {str(weights_path)}"
            )
            download_with_pbar(url, weights_path)

        self.__name = f"Metric3d_{config_path.name.split('.')[-2]}"
        self.__cfg = Metric3dConfig.fromfile(config_path)
        self.__model, _, _, _ = load_ckpt(
            weights_path,
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
