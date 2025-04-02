from enum import Enum
import itertools
from pathlib import Path
from typing import List, Optional

from gs_init_compare.depth_alignment.config import DepthAlignmentStrategyEnum
from gs_init_compare.depth_prediction.configs import Metric3dPreset


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PRESETS_DEPTH_SUBSAMPLE_FACTORS = [10, 20, "adaptive"]

MCMC_DEFAULT_PARAMS = {
    "init_opa": 0.5,
    "init_scale": 0.1,
    "opacity_reg": 0.01,
    "scale_reg": 0.01,
}


class Strategy(Enum):
    DEFAULT = "DefaultStrategy"
    MCMC = "MCMCStrategy"


def _make_depth_init_preset_base(
    model,
    subsample_factor,
    strategy: Strategy,
    noise_std_scene_frac: Optional[float],
    depth_alignment_strategy: DepthAlignmentStrategyEnum,
):
    additional_params = {Strategy.MCMC: MCMC_DEFAULT_PARAMS, Strategy.DEFAULT: {}}[
        strategy
    ]
    return {
        "strategy": strategy.value,
        "init_type": "monocular_depth",
        "mdi.predictor": model,
        "mdi.subsample_factor": subsample_factor,
        "mdi.pts_output_dir": "ply_export",
        "mdi.noise_std_scene_frac": noise_std_scene_frac,
        "mdi.depth_alignment_strategy": depth_alignment_strategy,
        **additional_params,
        # "mono_depth_pts_output_per_image": True,
        # "mono_depth_pts_only": True,
    }


def _make_metric3d_preset(*args):
    return {
        **_make_depth_init_preset_base("metric3d", *args),
        "mdi.metric3d.preset": Metric3dPreset.vit_large,
    }


def _make_unidepth_preset(*args):
    return {
        **_make_depth_init_preset_base("unidepth", *args),
        "mdi.unidepth.backbone": "vitl14",
    }


def _make_depthanything_v2_preset(model_type, *args):
    return {
        **_make_depth_init_preset_base("depth_anything_v2", *args),
        "mdi.depthanything.backbone": "vitl",
        "mdi.depthanything.model_type": model_type,
    }


# Not used for now since it requires EXIF data.
# def _make_depth_pro_preset(downsample_factor):
#     return {
#         **_make_depth_init_preset_base("depth_pro", downsample_factor),
#         "depth_pro_checkpoint": str(
#             PROJECT_ROOT / "third_party/apple_depth_pro/checkpoints/depth_pro.pt"
#         ),
#     }


def _make_moge_preset(*args):
    return _make_depth_init_preset_base("moge", *args)


ALL_NOISE_STD_SCENE_FRACTIONS = [None, 0.01, 0.05, 0.1, 0.15, 0.25]


def make_preset_name(
    name, downsample_factor, strategy, noise_std_scene_frac, depth_alignment_strategy
):
    name = f"{name}_depth_downsample_{downsample_factor}"
    if strategy != Strategy.DEFAULT:
        name += f"_{strategy.name.lower()}"
    if noise_std_scene_frac is not None:
        name += f"_noise_{noise_std_scene_frac}"
    if depth_alignment_strategy != DepthAlignmentStrategyEnum.lstsqrs:
        name += f"_{depth_alignment_strategy.value}"
    return name


def for_each_monodepth_setting_combination(
    depth_alignment_strategies: List[DepthAlignmentStrategyEnum] | None = None,
    downsample_factors: List[int | str] | None = None,
    noise_std_scene_fractions: List[float] | None = None,
    mcmc=False,
):
    if depth_alignment_strategies is None:
        depth_alignment_strategies = list(DepthAlignmentStrategyEnum)
    if downsample_factors is None:
        downsample_factors = PRESETS_DEPTH_SUBSAMPLE_FACTORS
    if noise_std_scene_fractions is None:
        noise_std_scene_fractions = [None]

    strategies = [Strategy.DEFAULT]
    if mcmc:
        strategies.append(Strategy.MCMC)

    for (
        downsample_factor,
        strategy,
        noise_std_scene_frac,
        depth_alignment_strategy,
    ) in itertools.product(
        downsample_factors,
        strategies,
        noise_std_scene_fractions,
        depth_alignment_strategies,
    ):
        args = (
            downsample_factor,
            strategy,
            noise_std_scene_frac,
            depth_alignment_strategy,
        )

        yield args


ALL_PREDICTOR_NAMES = [
    "metric3d",
    "moge",
    "unidepth",
    "depth_anything_v2_indoor",
    "depth_anything_v2_outdoor",
]


def make_presets(noise_std_scene_fractions=None) -> dict[str, dict]:
    if noise_std_scene_fractions is None:
        noise_std_scene_fractions = ALL_NOISE_STD_SCENE_FRACTIONS

    retval: dict[str, dict] = {
        "sfm": {},
        "sfm_mcmc": {"strategy": Strategy.MCMC.value, **MCMC_DEFAULT_PARAMS},
    }

    for args in for_each_monodepth_setting_combination(
        noise_std_scene_fractions=noise_std_scene_fractions,
        mcmc=True,
    ):
        retval[make_preset_name("metric3d", *args)] = _make_metric3d_preset(*args)
        retval[make_preset_name("moge", *args)] = _make_moge_preset(*args)
        retval[make_preset_name("unidepth", *args)] = _make_unidepth_preset(*args)
        for model_type in ["indoor", "outdoor"]:
            retval[make_preset_name(f"depth_anything_v2_{model_type}", *args)] = (
                _make_depthanything_v2_preset(model_type, *args)
            )

    return retval
