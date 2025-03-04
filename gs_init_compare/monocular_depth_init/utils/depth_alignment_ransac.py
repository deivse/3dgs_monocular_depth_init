from dataclasses import dataclass
import math
import torch


@dataclass
class DepthAlignmentParams:
    h: torch.Tensor

    @property
    def scale(self):
        return self.h[0]

    @property
    def shift(self):
        return self.h[1]


def align_depth_least_squares(depth, gt_depth):
    # Equations 2-5 in "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    # https://arxiv.org/pdf/1907.01341
    outer_product = torch.einsum("ib,jb->bij", depth, depth)
    h = torch.linalg.inv(torch.sum(outer_product, axis=0)
                         ) @ torch.sum(depth * gt_depth, axis=1)
    return DepthAlignmentParams(h)


def required_samples(
    inlier_count: int,
    total_correspondence_count: int,
    min_sample_size: int,
    confidence: float,
):
    # k = log(η)/ log(1 − P_I).
    # P_I ≈ ε^m
    inlier_ratio = inlier_count / total_correspondence_count
    try:
        return math.log(1 - confidence) / math.log(1 - inlier_ratio**min_sample_size)
    except (ZeroDivisionError, ValueError):
        return 0


def align_depth_ransac(
    depth: torch.Tensor,
    gt_depth: torch.Tensor,
    inlier_threshold: float,
    max_iters: int = 10000,
    confidence: float = 0.99,
) -> DepthAlignmentParams:
    SAMPLE_SIZE = 2
    num_samples = depth.shape[0]
    device = depth.device
    depth = torch.vstack(
        [
            depth.reshape(-1),
            torch.ones(num_samples, device=device),
        ]
    )

    h_best: DepthAlignmentParams | None = None
    inlier_indices_best = torch.zeros_like(gt_depth, dtype=bool)
    num_inliers_best = 0

    def calculate_inliers(h: DepthAlignmentParams):
        indices = torch.abs(
            h.scale * depth[0] + h.shift - gt_depth) ** 2 < inlier_threshold
        return indices, torch.sum(indices)

    if num_samples == 7398:
        print("Aligning depth using feedfA")

    for iteration in range(max_iters):
        sample_indices = torch.randint(0, num_samples, (SAMPLE_SIZE,))
        h = align_depth_least_squares(
            depth[:, sample_indices], gt_depth[sample_indices])
        inlier_indices, num_inliers = calculate_inliers(h)
        if num_inliers > num_inliers_best:
            # TODO: iterative local optimization?
            # https://cmp.felk.cvut.cz/ftp/articles/matas/chum-dagm03.pdf
            h_best = align_depth_least_squares(
                depth[:, inlier_indices], gt_depth[inlier_indices]
            )
            inlier_indices_best, num_inliers_best = calculate_inliers(h_best)

        if (
            required_samples(num_inliers_best, num_samples,
                             SAMPLE_SIZE, confidence)
            <= iteration
            and h_best is not None
        ):
            break

    h_best = align_depth_least_squares(
        depth[:, inlier_indices_best], gt_depth[inlier_indices_best]
    )
    inlier_indices_best = (
        torch.abs(h_best.scale * depth[0] +
                  h_best.shift - gt_depth) < inlier_threshold
    )
    num_inliers_best = torch.sum(inlier_indices_best)
    if math.isnan(h_best.scale) or math.isnan(h_best.shift) or math.isinf(h_best.scale) or math.isinf(h_best.shift):
        print(num_inliers_best, num_samples, h_best, iteration)
        raise ValueError(f"Alignment failed: {h_best}")

    #########################################
    h_test = align_depth_least_squares(depth, gt_depth)
    inlier_indices_test = (
        torch.abs(h_test.scale * depth[0] +
                  h_test.shift - gt_depth) < inlier_threshold
    )
    num_inliers_test = torch.sum(inlier_indices_test)
    avg_err_best = torch.mean(
        torch.abs(h_best.scale * depth[0] + h_best.shift - gt_depth)
    )
    avg_err_test = torch.mean(
        torch.abs(h_test.scale * depth[0] + h_test.shift - gt_depth)
    )
    print(
        f"#(iter): {iteration}, RANSAC inliers: {num_inliers_best}, Naive inliers: {num_inliers_test}, RANSAC Error: {avg_err_best}, Naive Error: {avg_err_test}"
    )
    ###########################################
    return h_best, num_inliers_best / num_samples
