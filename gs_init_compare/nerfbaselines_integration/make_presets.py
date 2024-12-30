from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PRESETS_DEPTH_DOWN_SAMPLE_FACTORS = [10, 20]


def _make_depth_init_preset_base(model, downsample_factor):
    return {
        "init_type": "monocular_depth",
        "mono_depth_model": model,
        "dense_depth_downsample_factor": downsample_factor,
        "mono_depth_pts_output_dir": "ply_export",
        # "mono_depth_pts_output_per_image": True,
        # "mono_depth_pts_only": True,
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


def _make_unidepth_preset(downsample_factor):
    return {
        **_make_depth_init_preset_base("unidepth", downsample_factor),
        "unidepth_backbone": "vitl14",
    }


def _make_depthanythingv2_preset(downsample_factor, model_type):
    return {
        **_make_depth_init_preset_base("depth_anything_v2", downsample_factor),
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


def _make_moge_preset(downsample_factor):
    return _make_depth_init_preset_base("moge", downsample_factor)


def make_presets() -> dict[str, dict]:
    retval: dict[str, dict] = {"sfm": {}}
    for downsample_factor in PRESETS_DEPTH_DOWN_SAMPLE_FACTORS:
        retval[f"metric3d_depth_downsample_{downsample_factor}"] = (
            _make_metric3d_preset(downsample_factor)
        )
        retval[f"moge_depth_downsample_{downsample_factor}"] = _make_moge_preset(
            downsample_factor
        )
        retval[f"unidepth_depth_downsample_{downsample_factor}"] = (
            _make_unidepth_preset(downsample_factor)
        )
        for model_type in ["indoor", "outdoor"]:
            retval[
                f"depth_anything_v2_{model_type}_depth_downsample_{downsample_factor}"
            ] = _make_depthanythingv2_preset(downsample_factor, model_type)

        # retval[f"depth_pro_depth_downsample_{downsample_factor}"] = (
        #     _make_depth_pro_preset(downsample_factor)
        # )
    return retval


def all_preset_names():
    return list(make_presets().keys())
