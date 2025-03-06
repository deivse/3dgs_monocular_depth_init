import torch
from .interface import DepthAlignmentParams, DepthAlignmentStrategy


def align_depth_least_squares(depth: torch.Tensor, gt_depth: torch.Tensor):
    """
    Args:
        depth: torch.Tensor of shape (2, N) 
               where N is the number of points, 
               the first row is the predicted depth and the second row is 1.
        gt_depth: torch.Tensor of shape (N,)
    """
    # Equations 2-5 in "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    # https://arxiv.org/pdf/1907.01341
    outer_product = torch.einsum("ib,jb->bij", depth, depth)
    h = torch.linalg.pinv(torch.sum(outer_product, axis=0)
                          ) @ torch.sum(depth * gt_depth, axis=1)
    return DepthAlignmentParams(h)


class DepthAlignmentLstSqrs(DepthAlignmentStrategy):
    @classmethod
    def estimate_alignment(cls, predicted_depth: torch.Tensor, gt_depth: torch.Tensor) -> DepthAlignmentParams:
        return align_depth_least_squares(
            torch.vstack(
                [
                    predicted_depth.flatten(),
                    torch.ones(predicted_depth.numel(),
                               device=predicted_depth.device),
                ]
            ), gt_depth
        )
