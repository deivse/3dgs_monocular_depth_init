import pickle

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from datasets.colmap import Parser
from scipy.ndimage import gaussian_filter

with open("rick.pkl", "rb") as f:
    inputs = pickle.load(f)


depth: np.ndarray = inputs["depth"]
camera_id: str = inputs["camera_id"]
image_name: str = inputs["image_name"]
parser: Parser = inputs["parser"]







####

cam2world = parser.camtoworlds[camera_id]
K = parser.Ks_dict[camera_id]
imsize = parser.imsize_dict[camera_id]

sfm_points = parser.points[parser.point_indices[image_name]]
sfm_point_err = parser.points_err[parser.point_indices[image_name]]

# plot3d(sfm_points)

R = np.linalg.inv(cam2world[:3, :3])
C = -cam2world[:3, :3] @ cam2world[:3, 3]

P = K @ R @ np.hstack([np.eye(3), -C[:, None]])

sfm_points_camera = P @ np.vstack([sfm_points.T, np.ones(sfm_points.shape[0])])
sfm_points_camera_homo = sfm_points_camera
sfm_points_camera = sfm_points_camera[:2] / sfm_points_camera[2]

valid_sfm_pt_indices = np.logical_and(
    np.logical_and(sfm_points_camera[0] >= 0, sfm_points_camera[0] < imsize[0]),
    np.logical_and(sfm_points_camera[1] >= 0, sfm_points_camera[1] < imsize[1]),
)
valid_sfm_pt_indices = np.logical_and(valid_sfm_pt_indices, sfm_points[:, 2] > 0)

valid_sfm_pt_indices = np.logical_and(
    valid_sfm_pt_indices, sfm_point_err < 1
)
print(f"{np.sum(valid_sfm_pt_indices)=}", len(valid_sfm_pt_indices))
sfm_points_camera = sfm_points_camera[:, valid_sfm_pt_indices]

depth_ratios = sfm_points_camera_homo[2, valid_sfm_pt_indices] / (
    1 + depth[sfm_points_camera[1].astype(int), sfm_points_camera[0].astype(int)]
)

depth_scalar = np.mean(depth_ratios)
print(f"{depth_scalar=}, {np.std(depth_ratios)=}")

###

camera_grid = np.dstack(
    [np.mgrid[0 : imsize[0], 0 : imsize[1]].T, depth_scalar * (1 + depth)]
)

# camera_grid[:, :, :2] *= parser.factor
camera_grid[:, :, 0] = camera_grid[:, :, 0] * camera_grid[:, :, 2]
camera_grid[:, :, 1] = camera_grid[:, :, 1] * camera_grid[:, :, 2]


def transform_camera_to_world_space(camera_homo, downsample_factor):
    dense_world = np.linalg.inv(K) @ camera_homo.reshape((-1, 3)).T
    dense_world = (cam2world @ np.vstack([dense_world, np.ones(dense_world.shape[1])]))[
        :3
    ].T
    dense_world = dense_world.reshape((imsize[0], imsize[1], 3))
    return dense_world[::downsample_factor, ::downsample_factor, :]


xyz_full = transform_camera_to_world_space(camera_grid, 1)
xyz = transform_camera_to_world_space(camera_grid, 10)
camera_plane_xyz = transform_camera_to_world_space(
    np.dstack([np.mgrid[0 : imsize[0], 0 : imsize[1]].T, np.ones(depth.shape)]),
    20,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Downsample xyz to include less points

plot3d(parser.points[::10,], "y", ax)
plot3d(sfm_points, "r", ax)
plot3d(xyz.reshape(-1, 3), "g", ax)
# plot3d(
#     xyz_full[
#         np.round(sfm_points_camera[0]).astype(int),
#         np.round(sfm_points_camera[1]).astype(int),
#         :,
#     ],
#     "g",
#     ax,
# )
plot3d(camera_plane_xyz, "b", ax)
plt.show()
