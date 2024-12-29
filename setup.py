from pathlib import Path
from setuptools import setup

# Hack for resolving some dependency paths relative to project root.
THIRD_PARTY_PATH = Path(__file__).parent / "gs_init_compare/third_party"

setup(
    install_requires=[
        f"depth_pro @ {(THIRD_PARTY_PATH / 'apple_depth_pro').as_uri()}",
        "unidepth @git+https://github.com/lpiccinelli-eth/UniDepth.git",
        "mmcv@https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl",
        "fused_ssim@git+https://github.com/rahul-goel/fused-ssim",
        "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl",
        "xformers~=0.0.24",
        "datetime~=5.5.0",
        "imagecorruptions~=1.1.2",
        "gradio-imageslider~=0.0.20",
        "cupy-cuda12x~=13.3.0",
        "nerfview~=0.0.3",
        "html4vision~=0.4.3",
        "imageio[ffmpeg]~=2.35.1",
        "pycolmap~=3.11",
        "microsoft-python-type-stubs @ git+https://github.com/microsoft/python-type-stubs.git",
        "ruff",
        "pillow",
        "nerfbaselines @ git+https://github.com/deivse/nerfbaselines.git@0b23fe2c65e1346e11d4a0eb62cf5afbb042bda4",
        "open3d~=0.18.0",
        # Moge requirements
        "click",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh",
        "pillow",
        "huggingface_hub",
        # utils3d is a dependency of MoGe that is included in their repo, but the imports are broken
        # so it's better to install it separately
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d",
    ]
)
