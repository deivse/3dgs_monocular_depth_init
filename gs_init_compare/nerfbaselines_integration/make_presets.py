from enum import Enum
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PRESETS_DEPTH_DOWN_SAMPLE_FACTORS = [10, 20, 30]


class Strategy(Enum):
    DEFAULT = "DefaultStrategy"
    MCMC = "MCMCStrategy"


def _make_depth_init_preset_base(
    model,
    downsample_factor,
    strategy: Strategy,
    noise_std_scene_frac: Optional[float],
):
    return {
        "init_type": "monocular_depth",
        "mono_depth_model": model,
        "dense_depth_downsample_factor": downsample_factor,
        "mono_depth_pts_output_dir": "ply_export",
        "strategy": strategy.value,
        "mono_depth_noise_std_scene_frac": noise_std_scene_frac,
        # "mono_depth_pts_output_per_image": True,
        # "mono_depth_pts_only": True,
    }


def _make_metric3d_preset(downsample_factor, strategy: Strategy, noise_std_scene_frac):
    return {
        **_make_depth_init_preset_base(
            "metric3d", downsample_factor, strategy, noise_std_scene_frac
        ),
        "metric3d_config": str(
            PROJECT_ROOT
            / "third_party/metric3d/mono/configs/HourglassDecoder/vit.raft5.large.py"
        ),
        # TODO: mechanism to download weights if missing... then don't use hardcoded path
        "metric3d_weights": "/workspaces/gs_init_comparison/metric3d_configs/metric_depth_vit_large_800k.pth",
    }


def _make_unidepth_preset(downsample_factor, strategy: Strategy, noise_std_scene_frac):
    return {
        **_make_depth_init_preset_base(
            "unidepth", downsample_factor, strategy, noise_std_scene_frac
        ),
        "unidepth_backbone": "vitl14",
    }


def _make_depthanything_v2_preset(
    downsample_factor, model_type, strategy: Strategy, noise_std_scene_frac
):
    return {
        **_make_depth_init_preset_base(
            "depth_anything_v2", downsample_factor, strategy, noise_std_scene_frac
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


def _make_moge_preset(downsample_factor, strategy: Strategy, noise_std_scene_frac):
    return _make_depth_init_preset_base(
        "moge", downsample_factor, strategy, noise_std_scene_frac
    )


ALL_NOISE_STD_SCENE_FRACTIONS = [None, 0.01, 0.05, 0.1, 0.15, 0.25]


def make_presets(noise_std_scene_fractions=None) -> dict[str, dict]:
    if noise_std_scene_fractions is None:
        noise_std_scene_fractions = ALL_NOISE_STD_SCENE_FRACTIONS

    retval: dict[str, dict] = {
        "sfm": {},
        # "sfm_mcmc": {"strategy": Strategy.MCMC.value}
    }
    for strategy in [Strategy.DEFAULT]:
        for downsample_factor in PRESETS_DEPTH_DOWN_SAMPLE_FACTORS:
            for noise_std_scene_frac in noise_std_scene_fractions:

                def _make_preset_name(name):
                    name = f"{name}_depth_downsample_{downsample_factor}"
                    if strategy != Strategy.DEFAULT:
                        name += f"{strategy.name.lower()}"
                    if noise_std_scene_frac is not None:
                        name += f"_noise_{noise_std_scene_frac}"
                    return name

                retval[_make_preset_name("metric3d")] = _make_metric3d_preset(
                    downsample_factor, strategy, noise_std_scene_frac
                )
                retval[_make_preset_name("moge")] = _make_moge_preset(
                    downsample_factor, strategy, noise_std_scene_frac
                )
                retval[_make_preset_name("unidepth")] = _make_unidepth_preset(
                    downsample_factor, strategy, noise_std_scene_frac
                )
                for model_type in ["indoor", "outdoor"]:
                    retval[_make_preset_name(f"depth_anything_v2_{model_type}")] = (
                        _make_depthanything_v2_preset(
                            downsample_factor,
                            model_type,
                            strategy,
                            noise_std_scene_frac,
                        )
                    )

                # retval[f"depth_pro_depth_downsample_{downsample_factor}"] = (
                #     _make_depth_pro_preset(downsample_factor)
                # )
    return retval


def get_preset_names(noise_std_scene_fractions):
    return list(make_presets(noise_std_scene_fractions).keys())
