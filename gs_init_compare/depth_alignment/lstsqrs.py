import torch
from .interface import DepthAlignmentStrategy


def align_depth_least_squares(depth: torch.Tensor, gt_depth: torch.Tensor):
    """
    Args:
        depth: torch.Tensor of shape (2, N)
               where N is the number of points,
               the first row is the predicted depth and the second row is 1.
        gt_depth: torch.Tensor of shape (N,)
    Returns:
        scale: float
        shift: float
    """
    # Equations 2-5 in "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    # https://arxiv.org/pdf/1907.01341
    outer_product = torch.einsum("ib,jb->bij", depth, depth)
    h = torch.linalg.pinv(torch.sum(outer_product, axis=0)) @ torch.sum(
        depth * gt_depth, axis=1
    )
    return h[0], h[1]


class DepthAlignmentLstSqrs(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        image: torch.Tensor,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        scale, shift = align_depth_least_squares(
            torch.vstack(
                [
                    predicted_depth[
                        sfm_points_camera_coords[1], sfm_points_camera_coords[0]
                    ].flatten(),
                    torch.ones(sfm_points_depth.numel(), device=predicted_depth.device),
                ]
            ),
            sfm_points_depth,
        )
        return predicted_depth * scale + shift
