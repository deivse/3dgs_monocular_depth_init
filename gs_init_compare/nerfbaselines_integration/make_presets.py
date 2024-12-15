from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEPTH_DOWN_SAMPLE_FACTORS = [10, 20]


def _make_depth_init_preset_base(model, downsample_factor):
    return {
        "init_type": "monocular_depth",
        "mono_depth_model": model,
        "dense_depth_downsample_factor": downsample_factor,
    }


def _make_metric3d_preset(downsample_factor):
    return {
        **_make_depth_init_preset_base("metric3d", downsample_factor),
        "metric3d_config": str(
            PROJECT_ROOT
            / "third_party/metric3d/mono/configs/HourglassDecoder/vit.raft5.large.py"
        ),
        # TODO: mechanism to download weights if missing... then don't use hardcoded path
        "metric3d_weights": "/workspaces/gs_init_comparison/metric3d_configs/metric_depth_vit_large_800k.pth",
    }


def _make_depth_pro_preset(downsample_factor):
    return {
        **_make_depth_init_preset_base("depth_pro", downsample_factor),
        "depth_pro_checkpoint": str(PROJECT_ROOT / "third_party/apple_depth_pro/checkpoints/depth_pro.pt"),
    }


def _make_moge_preset(downsample_factor):
    return _make_depth_init_preset_base("moge", downsample_factor)


def make_presets() -> dict[str, dict]:
    retval: dict[str, dict] = {
        "sfm": {}
    }
    for downsample_factor in DEPTH_DOWN_SAMPLE_FACTORS:
        retval[f"metric3d_depth_downsample_{downsample_factor}"] = _make_metric3d_preset(
            downsample_factor)
        retval[f"depth_pro_depth_downsample_{downsample_factor}"] = _make_depth_pro_preset(
            downsample_factor)
        retval[f"moge_depth_downsample_{downsample_factor}"] = _make_moge_preset(
            downsample_factor)
    return retval


def all_preset_names():
    return list(make_presets().keys())
