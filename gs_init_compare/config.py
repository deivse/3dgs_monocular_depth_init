from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union, Tuple, get_type_hints
from typing_extensions import assert_never

from gsplat.strategy import DefaultStrategy, MCMCStrategy

from gs_init_compare.point_cloud_postprocess.config import PointCloudPostprocessConfig


from .depth_alignment.config import DepthAlignmentStrategyEnum
from .depth_subsampling.config import AdaptiveSubsamplingConfig
from .depth_prediction.configs import (
    Metric3dV2Config,
    DepthAnythingV2Config,
    UnidepthConfig,
)


@dataclass
class MonocularDepthInitConfig:
    """
    Configuration of monocular depth initialization.
    """

    # Which monocular depth prediction model to use.
    predictor: Optional[
        Literal["metric3d", "depth_pro", "moge", "unidepth", "depth_anything_v2"]
    ] = "metric3d"

    metric3d: Metric3dV2Config = Metric3dV2Config()
    unidepth: UnidepthConfig = UnidepthConfig()
    depthanything: DepthAnythingV2Config = DepthAnythingV2Config()

    # Strategy to align predicted depth to depth of known SfM points.
    depth_alignment_strategy: DepthAlignmentStrategyEnum = (
        DepthAlignmentStrategyEnum.ransac
    )
    # How depth is subsampled to temper the number of generated 3D points.
    # If set to an int, a constant subsampling factor is used. If set to
    # "adaptive", adaptive subsampling is used, which can be further
    # configured using --mdi.adaptive-subsampling.
    subsample_factor: Union[int, Literal["adaptive"]] = 10
    # Configuration for adaptive subsampling. Ignored if not using "adaptive" subsampling.
    adaptive_subsampling: AdaptiveSubsamplingConfig = AdaptiveSubsamplingConfig()

    postprocess: PointCloudPostprocessConfig = PointCloudPostprocessConfig()

    # If set, point clouds from monocular depth init are saved to this directory.
    pts_output_dir: Optional[str] = None
    # If set, a point cloud is saved per-image, in addition to the final point cloud.
    pts_output_per_image: bool = False
    # If set, the program will exit after exporting point clouds.
    pts_only: bool = False

    # If set, normally distributed noise is added to the point cloud produced
    # by monocular depth initialization, with standard deviation equal to this fraction of the scene scale.
    noise_std_scene_frac: Optional[float] = None

    ignore_cache: bool = False
    cache_dir: str = "__mono_depth_cache__"


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Don't wait for user to close viewer after finishing training
    non_blocking_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_final_ply: bool = True

    # Initialization strategy
    init_type: Literal["sfm", "random", "monocular_depth"] = "sfm"

    mdi: MonocularDepthInitConfig = MonocularDepthInitConfig()

    # Initial number of GSs. Ignored if using sfm or monocular_depth
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_background: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Network used for Learned Perceptual Image Patch Similarity (LPIPS) loss
    lpips_net: Literal["vgg", "alex"] = "alex"

    # ====== nerfbaselines extensions ======

    # Appearance optimization eval settings
    app_test_opt_steps: int = 128
    app_test_opt_lr: float = 0.1

    # Background color for rendering
    background_color: Optional[Tuple[float, float, float]] = None

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)
