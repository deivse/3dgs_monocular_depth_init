from pathlib import Path
import numpy as np
import torch

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.lstsqrs import DepthAlignmentLstSqrs
from .interface import DepthAlignmentStrategy
from torchrbf import RBFInterpolator
import matplotlib.pyplot as plt


def align_depth_interpolate(
    depth: torch.Tensor,
    sfm_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    debug_export_dir: Path | None,
):
    """
    Args:
        depth: torch.Tensor of shape (Width, Height)
        sfm_points_camera_coords: torch.Tensor of shape (2, N)
                where N is the number of points, the first row is y and the second row is x.
        gt_depth: torch.Tensor of shape (N,)
    """
    H, W = depth.shape

    def rbf_interpolation(coords, values):
        interpolator = RBFInterpolator(
            coords,
            values,
            smoothing=0.001,
            kernel="thin_plate_spline",
            device=depth.device,
        )

        desired_width = 256
        factor = max(W / desired_width, 1)
        query_width = int(W / factor)
        query_height = int(H / factor)

        # Query coordinates
        x = torch.linspace(0, 1, query_width, device=depth.device)
        y = torch.linspace(0, 1, query_height, device=depth.device)
        grid_points = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack(grid_points, dim=-1).reshape(-1, 2)

        # Query RBF on grid points
        interpolated = interpolator(grid_points)
        interpolated = interpolated.reshape(query_width, query_height)[None, None, :, :]
        return torch.nn.functional.interpolate(
            interpolated,
            size=(W, H),
            mode="bilinear",
            align_corners=True,
        )[0, 0, :, :].T

    # first compute approximate offset using least squares
    lstsqrs_aligned = DepthAlignmentLstSqrs.align(
        predicted_depth=depth,
        sfm_points_camera_coords=sfm_points_camera_coords,
        sfm_points_depth=gt_depth,
    )

    if "9225" in str(debug_export_dir):
        print("BREAK")

    coords = torch.hstack(
        [
            (sfm_points_camera_coords[0] / (W - 1))[:, None],
            (sfm_points_camera_coords[1] / (H - 1))[:, None],
        ]
    )

    # limit number of points for RBF to avoid OOM
    MAX_RBF_POINTS = 5000
    if coords.shape[0] > MAX_RBF_POINTS:
        indices = torch.randperm(coords.shape[0], device=coords.device)[:MAX_RBF_POINTS]
        sfm_points_camera_coords = sfm_points_camera_coords[:, indices]
        coords = coords[indices]
        gt_depth = gt_depth[indices]

    # Compute interpolations
    rbf_scale = rbf_interpolation(
        coords,
        gt_depth
        / lstsqrs_aligned[sfm_points_camera_coords[1], sfm_points_camera_coords[0]],
    )

    # TODO: somehow get ransac-like outlier rejection while keeping the "adaptive alignment"?
    # TODO: study failure cases - when does this make things worse?
    # TODO: deal with negative depth values...

    aligned_depth = rbf_scale * lstsqrs_aligned

    if debug_export_dir is not None:
        print("Saving images to dir", debug_export_dir)

        vmin = min(lstsqrs_aligned.min(), aligned_depth.min()).item()
        vmax = max(lstsqrs_aligned.max(), aligned_depth.max()).item()
        debug_export_dir.mkdir(parents=True, exist_ok=True)

        # Save pre-RBF-alignment depth map
        # Create copies for visualization with red pixels where depth < 1
        lstsqrs_vis = lstsqrs_aligned.cpu().numpy().copy()
        aligned_vis = aligned_depth.cpu().numpy().copy()
        
        # Set pixels where depth < 1 to red (this will override the colormap)
        # We'll use a custom approach with RGB arrays
        import matplotlib.cm as cm
        
        # Normalize the depth values for colormap
        norm_lstsqrs = (lstsqrs_vis - vmin) / (vmax - vmin)
        norm_aligned = (aligned_vis - vmin) / (vmax - vmin)
        
        # Apply colormap
        lstsqrs_rgb = cm.plasma(norm_lstsqrs)
        aligned_rgb = cm.plasma(norm_aligned)
        
        # Set red color where depth < 0
        red_mask_lstsqrs = lstsqrs_vis < 0
        red_mask_aligned = aligned_vis < 0
        
        lstsqrs_rgb[red_mask_lstsqrs] = [1, 0, 0, 1]  # Red
        aligned_rgb[red_mask_aligned] = [1, 0, 0, 1]  # Red
        
        # Save pre-RBF-alignment depth map with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(lstsqrs_rgb)
        ax.set_title("Least Squares Aligned Depth")
        # Create a colorbar with the original depth values
        sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth")
        # Add note about red pixels
        ax.text(0.02, 0.98, "Red: depth < 0", transform=ax.transAxes, 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment='top')
        pre_rbf_path = Path(debug_export_dir) / "lstsqrs_depth.png"
        plt.savefig(pre_rbf_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save rbf scale with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        rbf_scale_np = rbf_scale.cpu().numpy()
        im = ax.imshow(rbf_scale_np, cmap="viridis")
        ax.set_title("RBF Scale Factor")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Scale Factor")
        rbf_scale_path = Path(debug_export_dir) / "rbf_scale.png"
        plt.savefig(rbf_scale_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save final aligned depth map with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(aligned_rgb)
        ax.set_title("RBF Aligned Depth")
        # Create a colorbar with the original depth values
        sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth")
        # Add note about red pixels
        ax.text(0.02, 0.98, "Red: depth < 0", transform=ax.transAxes, 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment='top')
        aligned_depth_path = Path(debug_export_dir) / "rbf_aligned_depth.png"
        plt.savefig(aligned_depth_path, dpi=150, bbox_inches='tight')
        plt.close()

    return aligned_depth


class DepthAlignmentInterpolate(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None,
    ) -> torch.Tensor:
        return align_depth_interpolate(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            debug_export_dir,
        )
