import torch

from gs_init_compare.point_cloud_postprocess.config import PointCloudPostprocessConfig

from kornia.color import rgb_to_grayscale
from kornia.feature import SIFTDescriptor

from gs_init_compare.types import InputImage

DESCRIPTOR_PATCH_SIZE = 32


def prepare_descriptors(
    postprocess_config: PointCloudPostprocessConfig,
    image: InputImage,
    subsampling_mask: torch.Tensor,
    device: str,
):
    # Mask out pixels closer than DESCRIPTOR_PATCH_SIZE/2 from border
    imsize = image.data.shape[:2]

    border = DESCRIPTOR_PATCH_SIZE // 2
    border_mask = torch.ones(imsize, dtype=bool)
    border_mask[:border, :] = False
    border_mask[-border:, :] = False
    border_mask[:, :border] = False
    border_mask[:, -border:] = False
    subsampling_mask = subsampling_mask & border_mask.view(-1)

    # Extract patches at all points included in subsampling mask
    patch_size = DESCRIPTOR_PATCH_SIZE
    half_patch = patch_size // 2
    indices = torch.nonzero(subsampling_mask.view(imsize), as_tuple=False)
    patches = []

    image_grayscale = rgb_to_grayscale(image.data.to(device).permute(2, 0, 1))

    patches = torch.empty(indices.shape[0], 1, patch_size, patch_size)
    for i, idx in enumerate(indices):
        x, y = idx[0].item(), idx[1].item()
        patch = image_grayscale[
            :, y - half_patch : y + half_patch, x - half_patch : x + half_patch
        ]
        patches[i] = patch

    SIFT = SIFTDescriptor(DESCRIPTOR_PATCH_SIZE, 8, 4)
    descriptors: torch.Tensor = SIFT(patches)
    return descriptors, subsampling_mask
