from nerfbaselines import register

_name = "gs-init-compare"

register(
    {
        "id": _name,
        "method_class": "gs_init_compare.nerfbaselines_integration.method:InitCompareGsplat",
        "conda": {
            "environment_name": "gs_init_compare_nerbaselines",
            "python_version": "3.10",
            "install_script": r"""
# TODO: Use https once the repo is public
git clone git@github.com:deivse/gs_init_comparison.git
cd gs_init_comparison
git fetch && git switch nerfbaselines

# TODO: pip install fails because ssl module is not available
./install.sh gs_init_compare_nerbaselines

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
        "required_features": frozenset(
            ("points3D_xyz", "points3D_rgb", "color", "images_points3D_indices")
        ),
        "supported_camera_models": frozenset(("pinhole",)),
        "supported_outputs": ("color", "depth", "accumulation"),
    }
)
