import numpy as np
import torch
import math
import pyexr
from util.visualize import visualize_point_list, visualize_grid

from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FromDistanceToDepth:

    def __init__(self, focal_length: float) -> None:
        self.focal_length = focal_length

    def __call__(self, distance_image: torch.tensor) -> torch.tensor:
        width = distance_image.shape[0]
        height = distance_image.shape[1]

        cx = width // 2
        cy = height // 2

        xs = torch.arange(width) - cx
        ys = torch.arange(height) - cy
        xis, yis = torch.meshgrid(xs, ys)

        depth = torch.sqrt(
            distance_image ** 2 / ((xis ** 2 + yis ** 2) / (self.focal_length ** 2) + 1)
        )

        return depth


def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max):
    x = image_size[0]
    y = image_size[1]
    eight_points = torch.tensor([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                             [0 * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, 0 * depth_max, depth_max, 1.0]]).transpose(1, 0)
    frustum = torch.mm(intrinsic_inv, eight_points)
    frustum = frustum.transpose(1, 0)
    return frustum[:, :3]


def coords_multiplication(A, B):
    B = np.concatenate([np.transpose(B), np.ones((1, B.shape[0]))])
    return np.transpose(np.dot(A, B))[:, :3]


def depth_to_camera(depth_map, f, cx, cy):
    #if depth_map.size() ## TO-DO: enable batching and non-batching mode. Currently only supports batched inputs
    bs = depth_map.shape[0]
    v, u = torch.meshgrid(torch.arange(depth_map.shape[-2], device=depth_map.device), torch.arange(depth_map.shape[-1], device=depth_map.device))
    X = ((torch.multiply(u, depth_map) - cx * depth_map) / f)
    Y = -((torch.multiply(v, depth_map) - cy * depth_map) / f)
    Z = depth_map
    #return X.flatten(), Y.flatten(), Z.flatten()
    return X.reshape((bs, -1)), Y.reshape((bs, -1)), Z.reshape((bs, -1))


def generate_frustum_volume(frustum, voxelsize):
    maxx = torch.max(frustum[:, 0]) / voxelsize
    maxy = torch.max(frustum[:, 1]) / voxelsize
    maxz = torch.max(frustum[:, 2]) / voxelsize
    minx = torch.min(frustum[:, 0]) / voxelsize
    miny = torch.min(frustum[:, 1]) / voxelsize
    minz = torch.min(frustum[:, 2]) / voxelsize

    dimX = torch.ceil(maxx - minx)
    dimY = torch.ceil(maxy - miny)
    dimZ = torch.ceil(maxz - minz)
    camera2frustum = torch.tensor([[1.0 / voxelsize, 0, 0, -minx],
                               [0, 1.0 / voxelsize, 0, -miny],
                               [0, 0, 1.0 / voxelsize, -minz],
                               [0, 0, 0, 1.0]], device = device)

    return (dimX, dimY, dimZ), camera2frustum


def depth_to_gridspace(distance_map, intrinsic_path=None, read=True):
    # read depth from path
    if read:
        # a matrix of (height x width x channels)
        input_depth = pyexr.open(distance_map).get("R")[:, :, 0]
    else:
        input_depth = distance_map
    # read intrinsics
    if intrinsic_path is None:
        intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")

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
    depth_in_gridspace = depth_in_gridspace.view(distance_map.shape[0], -1)
    return depth_in_gridspace

def depthmap_to_gridspace(depthmap):
    intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")
    intrinsic = get_intrinsic(intrinsic_path)
    focal_length, cx, cy = intrinsic[0][0], intrinsic[0][2], intrinsic[1][2]

    # depth to camera space
    X, Y, Z = depth_to_camera(depthmap, focal_length, cx, cy)

    # get camera space to grid space transform
    intrinsic_inv = torch.inverse(intrinsic)
    frustum = generate_frustum([320, 240], intrinsic_inv, 0.4, 6.0)
    dims, camera2frustum = generate_frustum_volume(frustum, 0.05)

    # depth from camera to grid space
    camera2frustum = camera2frustum.expand((X.shape[0],-1, -1))
    coords = torch.stack([X, Y, Z, torch.ones_like(X)]).transpose(1,0)
    #print(camera2frustum.shape, coords.shape)
    depth_in_gridspace = (camera2frustum @ coords)[:,:3, :].transpose(-1,-2)
    return depth_in_gridspace

def get_intrinsic(intrinsic_path):
    intrinsic_line_0, intrinsic_line_1 = intrinsic_path.read_text().splitlines()[:2]
    focal_length = float(intrinsic_line_0[2:].split(',')[0])
    cx = float(intrinsic_line_0[2:-2].split(',')[2].strip())
    cy = float(intrinsic_line_1[1:-2].split(',')[2].strip())
    intrinsic = torch.tensor([[focal_length, 0, cx, 0], [0, focal_length, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return intrinsic


if __name__ == "__main__":
    from pathlib import Path
    device = torch.device("cpu")

    distance_map_path = str(Path("data") / "raw" / "overfit" / "00000" / "distance.exr")
    depth_map_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth_map.exr"
    intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")
    output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "depthmap_to_pc.obj"

    #depth_grid_space = depth_to_gridspace(distance_map_path, intrinsic_path)
    depth_map = pyexr.open(str(depth_map_path)).get("Z")[:,:,0]
    depth_map = torch.from_numpy(depth_map)
    pointcloud = depthmap_to_gridspace(depth_map)
    pointcloud = pointcloud.reshape(-1,3).squeeze()

    #visualize_point_list(depth_grid_space, output_pt_cloud_path)
    visualize_point_list(pointcloud, output_pt_cloud_path)

    dims = (139, 104, 112)
    # visualize as voxels
    #output_voxel_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth_voxels.obj"
    #grid = np.zeros(dims)
    #to_int = lambda x: np.round(x).astype(np.int32)
    #grid[to_int(depth_grid_space[:, 0]), to_int(depth_grid_space[:, 1]), to_int(depth_grid_space[:, 2])] = 1
    #visualize_grid(grid, output_voxel_path)

    # lets also visualize in occupancy space, which is just normalized grid space

    # center
    #depth_grid_space[:, 0] -= (dims[0] / 2)
    #depth_grid_space[:, 1] -= (dims[1] / 2)
    #depth_grid_space[:, 2] -= (dims[2] / 2)

    # scale
    #max_dim = np.array(dims).max()
    #depth_grid_space /= max_dim

    #output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "depth_occupied.obj"
    #visualize_point_list(depth_grid_space, output_pt_cloud_path)
