from pathlib import Path
from typing import FrozenSet, Optional, Union, List
import os
import logging

import numpy as np
from PIL import Image

from nerfbaselines import (
    DatasetNotFoundError,
    new_dataset,
    CameraModel,
    camera_model_to_int,
    DatasetFeature,
    new_cameras,
)
from nerfbaselines.datasets._colmap_utils import (
    read_points3D_binary,
    read_points3D_text,
    read_images_binary,
    read_images_text,
)
from nerfbaselines.datasets.nerfstudio import (
    MAX_AUTO_RESOLUTION,
    load_from_json,
    CAMERA_MODEL_TO_TYPE,
    _downscale_cameras,
)

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args


def customized_load_nerfstudio_dataset(
    path: Union[Path, str],
    split: str,
    downscale_factor: Optional[int] = None,
    features: Optional[FrozenSet[DatasetFeature]] = None,
    **kwargs,
):
    del kwargs
    path = Path(path)
    downscale_factor_original = downscale_factor
    downscale_factor = None

    def _get_fname(
        filepath: Path, data_dir: Path, downsample_folder_prefix="images_"
    ) -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """
        nonlocal downscale_factor

        if downscale_factor is None:
            if downscale_factor_original is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (
                        data_dir
                        / f"{downsample_folder_prefix}{2 ** (df + 1)}"
                        / filepath.name
                    ).exists():
                        break
                    df += 1

                downscale_factor = 2**df
                logging.info(f"Auto image downscale factor of {downscale_factor}")
            else:
                downscale_factor = downscale_factor_original

        # pyright workaround
        assert downscale_factor is not None

        if downscale_factor > 1:
            return (
                data_dir
                / f"{downsample_folder_prefix}{downscale_factor}"
                / filepath.name
            )
        return data_dir / filepath

    assert path.exists(), f"Data directory {path} does not exist."

    if path.suffix == ".json":
        meta = load_from_json(path)
        data_dir = path.parent
    elif (path / "transforms.json").exists():
        meta = load_from_json(path / "transforms.json")
        data_dir = path
    else:
        raise DatasetNotFoundError(f"Could not find transforms.json in {path}")

    def _process_split(split_frames_key):
        image_filenames: List[str] = []
        mask_filenames: List[str] = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta[split_frames_key]:
            filepath = Path(frame["file_path"])
            fname = _get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta[split_frames_key][ind] for ind in inds]
        assert downscale_factor is not None, "downscale_factor should be set by now"

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = _get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    np.array(
                        [
                            float(frame["k1"]) if "k1" in frame else 0.0,
                            float(frame["k2"]) if "k2" in frame else 0.0,
                            float(frame["p1"]) if "p1" in frame else 0.0,
                            float(frame["p2"]) if "p2" in frame else 0.0,
                            float(frame["k3"]) if "k3" in frame else 0.0,
                            float(frame["k4"]) if "k4" in frame else 0.0,
                        ],
                        dtype=np.float32,
                    )
                )

            image_filenames.append(str(fname))
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = _get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(str(mask_fname))

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = _get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        poses = np.array(poses).astype(np.float32)

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE.get(meta.get("camera_model"))
            if camera_type is None or camera_type not in get_args(CameraModel):
                raise NotImplementedError(
                    f"Camera model {meta.get('camera_model')} is not supported."
                )
        else:
            if distort_fixed:
                has_distortion = any(
                    meta[x] != 0.0 for x in ["k1", "k2", "p1", "p2", "k3", "k4"]
                )
            else:
                has_distortion = any(np.any(x != 0.0) for x in distort)
            camera_type = "opencv" if has_distortion else "pinhole"

        fx = (
            np.full((len(poses),), meta["fl_x"], dtype=np.float32)
            if fx_fixed
            else np.array(fx, dtype=np.float32)
        )
        fy = (
            np.full((len(poses),), meta["fl_y"], dtype=np.float32)
            if fy_fixed
            else np.array(fy, dtype=np.float32)
        )
        cx = (
            np.full((len(poses),), meta["cx"], dtype=np.float32)
            if cx_fixed
            else np.array(cx, dtype=np.float32)
        )
        cy = (
            np.full((len(poses),), meta["cy"], dtype=np.float32)
            if cy_fixed
            else np.array(cy, dtype=np.float32)
        )
        height = (
            np.full((len(poses),), meta["h"], dtype=np.int32)
            if height_fixed
            else np.array(height, dtype=np.int32)
        )
        width = (
            np.full((len(poses),), meta["w"], dtype=np.int32)
            if width_fixed
            else np.array(width, dtype=np.int32)
        )
        if distort_fixed:
            distortion_params = np.repeat(
                np.array(
                    [
                        float(meta["k1"]) if "k1" in meta else 0.0,
                        float(meta["k2"]) if "k2" in meta else 0.0,
                        float(meta["p2"]) if "p1" in meta else 0.0,
                        float(meta["p1"]) if "p2" in meta else 0.0,
                        float(meta["k3"]) if "k3" in meta else 0.0,
                        float(meta["k4"]) if "k4" in meta else 0.0,
                    ]
                )[None, :],
                len(poses),
                0,
            )
        else:
            distortion_params = np.stack(distort, 0)

        c2w = poses[:, :3, :4]

        # Convert from OpenGL to OpenCV coordinate system
        c2w[..., 0:3, 1:3] *= -1

        all_cameras = new_cameras(
            poses=c2w.astype(np.float32),
            intrinsics=np.stack([fx, fy, cx, cy], -1).astype(np.float32),
            camera_models=np.full(
                (len(poses),), camera_model_to_int(camera_type), dtype=np.uint8
            ),
            distortion_parameters=distortion_params.astype(np.float32),
            image_sizes=np.stack([width, height], -1).astype(np.int32),
            nears_fars=None,
        )

        # transform_matrix = torch.eye(4, dtype=torch.float32)
        # scale_factor = 1.0
        # if "applied_transform" in meta:
        #     applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        #     transform_matrix = transform_matrix @ torch.cat(
        #         [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
        #     )
        #     transform_matrix
        # if "applied_scale" in meta:
        #     applied_scale = float(meta["applied_scale"])
        #     scale_factor *= applied_scale
        if downscale_factor > 1:
            images_root = data_dir / f"images_{downscale_factor}"
            masks_root = data_dir / f"masks_{downscale_factor}"
            all_cameras = _downscale_cameras(all_cameras, downscale_factor)
        else:
            images_root = data_dir
            masks_root = data_dir

        # "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
        # "depth_unit_scale_factor": depth_unit_scale_factor,

        points3D_rgb = None
        points3D_xyz = None
        images_points3D_indices: Optional[List[np.ndarray]] = None
        if "points3D_xyz" in (features or {}):
            colmap_path = data_dir / "colmap" / "sparse" / "0"
            if not colmap_path.exists():
                colmap_path = data_dir / "sparse" / "0"
            if not colmap_path.exists():
                colmap_path = data_dir / "sparse"
            if not colmap_path.exists():
                colmap_path = data_dir
            points3D = None
            if (colmap_path / "points3D.bin").exists():
                points3D = read_points3D_binary(str(colmap_path / "points3D.bin"))
            elif (colmap_path / "points3D.txt").exists():
                points3D = read_points3D_text(str(colmap_path / "points3D.txt"))
            if points3D is not None:
                points3D_xyz = np.array(
                    [p.xyz for p in points3D.values()], dtype=np.float32
                )
                points3D_rgb = np.array(
                    [p.rgb for p in points3D.values()], dtype=np.uint8
                )

                # Transform xyz to match nerfstudio loader
                points3D_xyz = points3D_xyz[..., np.array([1, 0, 2])]
                points3D_xyz[..., 2] *= -1

                if "images_points3D_indices" in (features or {}):
                    points3D_map = {k: i for i, k in enumerate(points3D.keys())}
                    if (colmap_path / "points3D.bin").exists():
                        images_colmap = read_images_binary(
                            str(colmap_path / "images.bin")
                        )
                    elif (colmap_path / "points3D.txt").exists():
                        images_colmap = read_images_text(
                            str(colmap_path / "images.txt")
                        )
                    else:
                        raise RuntimeError(
                            f"3D points are requested but images.{{bin|txt}} not present in dataset {data_dir}"
                        )
                    images_colmap_map = {}
                    for image in images_colmap.values():
                        # Point3D ID is -1 for keypoints without corresponding 3D points
                        images_colmap_map[image.name] = np.array(
                            [points3D_map[x] for x in image.point3D_ids if x != -1],
                            dtype=np.int32,
                        )
                    images_points3D_indices = []
                    for impath in image_filenames:
                        impath = os.path.relpath(impath, str(images_root))
                        images_points3D_indices.append(images_colmap_map[impath])

        print(f"Loaded {len(image_filenames)} images for split {split}.")
        return new_dataset(
            cameras=all_cameras,
            image_paths=image_filenames,
            image_paths_root=str(images_root),
            mask_paths=mask_filenames if len(mask_filenames) > 0 else None,
            mask_paths_root=str(masks_root) if len(mask_filenames) > 0 else None,
            points3D_xyz=points3D_xyz,
            points3D_rgb=points3D_rgb,
            images_points3D_indices=images_points3D_indices,
            metadata={
                "color_space": "srgb",
                "type": None,
                "evaluation_protocol": "default",
                "downscale_factor": downscale_factor if downscale_factor > 1 else None,
            },
        )

    if split == "train":
        return _process_split("frames")
    elif split == "test":
        return _process_split("test_frames")
    else:
        raise ValueError(f"Unknown split {split}")
