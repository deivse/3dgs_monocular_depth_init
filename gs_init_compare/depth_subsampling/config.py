from dataclasses import dataclass


@dataclass
class DepthSubsamplingConfig:
    """
    Configures which heuristics to use for adaptive subsampling.
    """

    # If set, areas with stronger maximum color gradient will receive more samples.
    use_color_grad: bool = False
    # If set, depth compensation is used, such that surfaces in 3D receive a similar
    # numbers of initialization points, independent of their distance to the camera.
    use_depth: bool = True
    # If set, surface angle compensation is used, such that surfaces in 3D receive a similar
    # numbers of initialization points, independent of the angle of the surface normal and the
    # view direction. Surface normal estimation is based on predicted depth maps
    # and may not be precise.
    use_surface_angle: bool = False
