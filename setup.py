from pathlib import Path
from setuptools import setup

# Hack for resolving some dependency paths relative to project root.
THIRD_PARTY_PATH = Path(__file__).parent / "gs_init_compare/third_party"
NATIVE_MODULES_PATH = Path(__file__).parent / "native_modules"

setup(
    install_requires=[
        f"mdi_native_modules @ {NATIVE_MODULES_PATH.as_uri()}",
        f"depth_pro @ {(THIRD_PARTY_PATH / 'apple_depth_pro').as_uri()}",
        "segment_anything @git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf",
        "mmcv@https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl",
        "fused_ssim@git+https://github.com/rahul-goel/fused-ssim@30fb258c8a38fe61e640c382f891f14b2e8b0b5a",
        "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.2/gsplat-1.5.2%2Bpt22cu121-cp310-cp310-linux_x86_64.whl",
        "xformers==0.0.24",
        "transformers==4.49.0",
        "datetime==5.5.0",
        "imagecorruptions==1.1.2",
        "gradio-imageslider==0.0.20",
        "cupy-cuda12x==13.3.0",
        "nerfview==0.0.3",
        "html4vision==0.4.3",
        "imageio[ffmpeg]==2.35.1",
        "pycolmap==3.11",
        "microsoft-python-type-stubs @ git+https://github.com/microsoft/python-type-stubs.git",
        "pillow==10.2.0",
        "nerfbaselines @ git+https://github.com/deivse/nerfbaselines.git@patch_eval",
        "open3d==0.18.0",
        "filelock==3.19.1",
        "torchrbf==1.0.0",
        "huggingface_hub==0.34.4",
        # Moge requirements
        "click==8.2.1",
        "opencv-python==4.10.0",
        "scipy==1.15.2",
        "matplotlib==3.9.2",
        "trimesh==4.4.9",
        # utils3d is a dependency of MoGe that is included in their repo, but the imports are broken
        # so it's better to install it separately
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@db0c68cbdc9a26bea1a580a2e39c6b8ff12f87b5",
        # Unidepth requirements since their torchhub thing is not completely without dependencies.
        "einops==0.8.1",
        "wandb==0.22.3",
    ],
)
