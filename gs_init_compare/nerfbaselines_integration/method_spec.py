import os
from nerfbaselines import register
from pathlib import Path


_name = "gs-init-compare"


PROJECT_ROOT = Path(__file__).parent.parent.absolute()

register(
    {
        "id": _name,
        "method_class": "gs_init_compare.nerfbaselines_integration.method:InitCompareGsplat",
        "conda": {
            "environment_name": "gs_init_compare",
            "python_version": "3.10",
            "install_script": r"""
# TODO: Use https once the repo is public
git clone git@github.com:deivse/gs_init_comparison.git
cd gs_init_comparison

./install.sh

# Clear build dependencies
if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
""",
        },
        "metadata": {
            "name": "depthinit_gsplat",
            "description": """TODO""",
        },
        "presets": {
            "sfm": {},
            "metric3d_vit_large": {
                "init_type": "monocular_depth",
                "mono_depth_model": "metric3d",
                "metric3d_config": str(
                    PROJECT_ROOT
                    / "third_party/metric3d/mono/configs/HourglassDecoder/vit.raft5.large.py"
                ),
                # TODO: mechanism to download weights if missing... then don't use hardcoded path
                "metric3d_weights": "/workspaces/gs_init_comparison/metric3d_configs/metric_depth_vit_large_800k.pth",
            },
            "blender": {
                "@apply": [{"dataset": "blender"}],
                "init_type": "random",
                "background_color": (1.0, 1.0, 1.0),
                "init_extent": 0.5,
            },
            "phototourism": {
                "@apply": [{"dataset": "phototourism"}],
                "app_opt": True,  # Enable appearance optimization
                "steps_scaler": 3.333334,  # 100k steps
            },
        },
        "implementation_status": {
            "mipnerf360": "reproducing",
            "blender": "working",
            "tanksandtemples": "working",
            "phototourism": "working",
        },
    }
)
