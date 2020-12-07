import numpy as np
import math
import pyexr
from util.visualize import visualize_point_list, visualize_grid


class FromDistanceToDepth:

    def __init__(self, focal_length: float) -> None:
        self.focal_length = focal_length

    def __call__(self, distance_image: np.array) -> np.array:
        width = distance_image.shape[0]
        height = distance_image.shape[1]

        cx = width // 2
        cy = height // 2

        xs = np.arange(width) - cx
        ys = np.arange(height) - cy
        xis, yis = np.meshgrid(ys, xs)

        depth = np.sqrt(
            distance_image ** 2 / ((xis ** 2 + yis ** 2) / (self.focal_length ** 2) + 1)
        )

        return depth


def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max):
    x = image_size[0]
    y = image_size[1]
    eight_points = np.array([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                             [0 * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, 0 * depth_max, depth_max, 1.0]]).transpose()
    frustum = np.dot(intrinsic_inv, eight_points)
    frustum = frustum.transpose()
    return frustum[:, :3]


def coords_multiplication(A, B):
    B = np.concatenate([np.transpose(B), np.ones((1, B.shape[0]))])
    return np.transpose(np.dot(A, B))[:, :3]


def depth_to_camera(depth_map, f, cx, cy):
    u, v = np.meshgrid(list(range(depth_map.shape[1])), list(range(depth_map.shape[0])))
    X = ((np.multiply(u, depth_map) - cx * depth_map) / f)
    Y = -((np.multiply(v, depth_map) - cy * depth_map) / f)
    Z = depth_map
    return X.flatten(), Y.flatten(), Z.flatten()


def generate_frustum_volume(frustum, voxelsize):
    maxx = np.max(frustum[:, 0]) / voxelsize
    maxy = np.max(frustum[:, 1]) / voxelsize
    maxz = np.max(frustum[:, 2]) / voxelsize
    minx = np.min(frustum[:, 0]) / voxelsize
    miny = np.min(frustum[:, 1]) / voxelsize
    minz = np.min(frustum[:, 2]) / voxelsize

    dimX = math.ceil(maxx - minx)
    dimY = math.ceil(maxy - miny)
    dimZ = math.ceil(maxz - minz)
    camera2frustum = np.array([[1.0 / voxelsize, 0, 0, -minx],
                               [0, 1.0 / voxelsize, 0, -miny],
                               [0, 0, 1.0 / voxelsize, -minz],
                               [0, 0, 0, 1.0]])

    return (dimX, dimY, dimZ), camera2frustum


def depth_to_gridspace(distance_map_path, intrinsic_path):
    # read depth
    input_depth = pyexr.open(distance_map_path).get("R")[:, :, 0]
    # read intrinsics
    intrinsic_line_0, intrinsic_line_1 = intrinsic_path.read_text().splitlines()[:2]
    focal_length = float(intrinsic_line_0[2:].split(',')[0])
    cx = float(intrinsic_line_0[2:-2].split(',')[2].strip())
    cy = float(intrinsic_line_1[1:-2].split(',')[2].strip())
    intrinsic = np.array([[focal_length, 0, cx, 0], [0, focal_length, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # distance map to depth map
    transform = FromDistanceToDepth(focal_length)
    depth_image = transform(input_depth)

    # Transforming depth map to grid space (in which the mesh resides)

    # depth to camera space
    X, Y, Z = depth_to_camera(depth_image, focal_length, cx, cy)

    # get camera space to grid space transform
    intrinsic_inv = np.linalg.inv(intrinsic)
    frustum = generate_frustum([320, 240], intrinsic_inv, 0.4, 6.0)
    dims, camera2frustum = generate_frustum_volume(frustum, 0.05)

    # depth from camera to grid space
    depth_in_gridspace = (camera2frustum @ np.stack([X, Y, Z, np.ones_like(X)]))[:3, :].T
    return depth_in_gridspace


if __name__ == "__main__":
    from pathlib import Path

    distance_map_path = str(Path("data") / "raw" / "overfit" / "00000" / "distance_0010.exr")
    intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")
    output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth.obj"

    depth_grid_space = depth_to_gridspace(distance_map_path, intrinsic_path)

    visualize_point_list(depth_grid_space, output_pt_cloud_path)

    dims = (139, 104, 112)
    # visualize as voxels
    output_voxel_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth_voxels.obj"
    grid = np.zeros(dims)
    to_int = lambda x: np.round(x).astype(np.int32)
    grid[to_int(depth_grid_space[:, 0]), to_int(depth_grid_space[:, 1]), to_int(depth_grid_space[:, 2])] = 1
    visualize_grid(grid, output_voxel_path)

    # lets also visualize in occupancy space, which is just normalized grid space

    # center
    depth_grid_space[:, 0] -= (dims[0] / 2)
    depth_grid_space[:, 1] -= (dims[1] / 2)
    depth_grid_space[:, 2] -= (dims[2] / 2)

    # scale
    max_dim = np.array(dims).max()
    depth_grid_space /= max_dim

    output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth_occupied.obj"
    visualize_point_list(depth_grid_space, output_pt_cloud_path)
