from enum import Enum
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PRESETS_DEPTH_DOWN_SAMPLE_FACTORS = [10, 20, 30]


class Strategy(Enum):
    DEFAULT = "DefaultStrategy"
    MCMC = "MCMCStrategy"


def _make_depth_init_preset_base(model, downsample_factor, strategy: Strategy):
    return {
        "init_type": "monocular_depth",
        "mono_depth_model": model,
        "dense_depth_downsample_factor": downsample_factor,
        "mono_depth_pts_output_dir": "ply_export",
        "strategy": strategy.value,
        # "mono_depth_pts_output_per_image": True,
        # "mono_depth_pts_only": True,
    }


def _make_metric3d_preset(downsample_factor, strategy: Strategy):
    return {
        **_make_depth_init_preset_base("metric3d", downsample_factor, strategy),
        "metric3d_config": str(
            PROJECT_ROOT
            / "third_party/metric3d/mono/configs/HourglassDecoder/vit.raft5.large.py"
        ),
        # TODO: mechanism to download weights if missing... then don't use hardcoded path
        "metric3d_weights": "/workspaces/gs_init_comparison/metric3d_configs/metric_depth_vit_large_800k.pth",
    }


def _make_unidepth_preset(downsample_factor, strategy: Strategy):
    return {
        **_make_depth_init_preset_base("unidepth", downsample_factor, strategy),
        "unidepth_backbone": "vitl14",
    }


def _make_depthanythingv2_preset(downsample_factor, model_type, strategy: Strategy):
    return {
        **_make_depth_init_preset_base(
            "depth_anything_v2", downsample_factor, strategy
        ),
        "depth_anything_backbone": "vitl",
        "depth_anything_model_type": model_type,
    }


# Not used for now since it requires EXIF data.
# def _make_depth_pro_preset(downsample_factor):
#     return {
#         **_make_depth_init_preset_base("depth_pro", downsample_factor),
#         "depth_pro_checkpoint": str(
#             PROJECT_ROOT / "third_party/apple_depth_pro/checkpoints/depth_pro.pt"
#         ),
#     }


def _make_moge_preset(downsample_factor, strategy: Strategy):
    return _make_depth_init_preset_base("moge", downsample_factor, strategy)


def make_presets() -> dict[str, dict]:
    retval: dict[str, dict] = {
        "sfm": {},
        # "sfm_mcmc": {"strategy": Strategy.MCMC.value}
    }
    for strategy in [Strategy.DEFAULT]:
        for downsample_factor in PRESETS_DEPTH_DOWN_SAMPLE_FACTORS:

            def _make_preset_name(name):
                if strategy == Strategy.DEFAULT:
                    return f"{name}_depth_downsample_{downsample_factor}"
                else:
                    return f"{name}_depth_downsample_{downsample_factor}_{strategy.name.lower()}"

            retval[_make_preset_name("metric3d")] = _make_metric3d_preset(
                downsample_factor, strategy
            )
            retval[_make_preset_name("moge")] = _make_moge_preset(
                downsample_factor, strategy
            )
            retval[_make_preset_name("unidepth")] = _make_unidepth_preset(
                downsample_factor, strategy
            )
            for model_type in ["indoor", "outdoor"]:
                retval[_make_preset_name(f"depth_anything_v2_{model_type}")] = (
                    _make_depthanythingv2_preset(
                        downsample_factor, model_type, strategy
                    )
                )

            # retval[f"depth_pro_depth_downsample_{downsample_factor}"] = (
            #     _make_depth_pro_preset(downsample_factor)
            # )
    return retval


def all_preset_names():
    return list(make_presets().keys())
