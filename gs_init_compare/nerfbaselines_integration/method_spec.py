import os
from nerfbaselines import register
from pathlib import Path


_name = "depthinit_gsplat"


PROJECT_ROOT = Path(__file__).parent.parent.absolute()

register(
    {
        "id": _name,
        "method_class": "gs_init_compare.nerfbaselines_integration.method:DepthInitGsplat",
        "conda": {
            "environment_name": _name,
            "python_version": "3.10",
            "install_script": r"""
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git checkout cc800d7750a891ab8684eac4ddbcf90b79d16295
git submodule init
git submodule update --recursive
conda develop "$PWD/examples"

# Install build dependencies
conda install -y cuda-toolkit 'numpy<2.0.0' pytorch==2.1.2 torchvision==0.16.2 -c pytorch -c nvidia/label/cuda-11.8.0
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
if [ "$NERFBASELINES_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi

# Install dependencies
pip install opencv-python-headless==4.10.0.84 -r examples/requirements.txt plyfile==0.8.1

# Install and build gsplat
pip install -e . --use-pep517 --no-build-isolation
python -c 'from gsplat import csrc'  # Test import

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
