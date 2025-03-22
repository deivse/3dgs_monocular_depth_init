from dataclasses import dataclass
from typing import Sequence


@dataclass
class AdaptiveSubsamplingConfig:
    """
    Configures which heuristics to use for adaptive subsampling.
    """

    # List of possible subsample factors to choose from.
    factors: Sequence[int] = (5, 10, 15)

    # The image is divided into tiles, a constant subsampling factor is used on each tile.
    # This parameter controls the size of these tiles.
    tile_size: int = 20

    # If set, depth compensation is used, such that surfaces in 3D receive a similar
    # number of initialization points, independent of their distance to the camera.
    use_depth: bool = True
    # If set, surface angle compensation is used, such that surfaces in 3D receive a similar
    # number of initialization points, independent of the angle of the surface normal and the
    # view direction. Surface normal estimation is based on predicted depth maps
    # and may not be precise.
    use_surface_angle: bool = False
    # If set, areas with stronger maximum color gradient will receive more samples.
    use_color_grad: bool = False
