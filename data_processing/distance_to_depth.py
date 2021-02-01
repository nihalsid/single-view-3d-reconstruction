import torch
import pyexr
from util.visualize import visualize_point_list, visualize_grid
from pathlib import Path

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
    frustum = (torch.mm(intrinsic_inv, eight_points)).transpose(1, 0)
    
    return frustum[:, :3]
    

def depth_to_camera(depth_map, f, cx, cy):
    v, u = torch.meshgrid(torch.arange(depth_map.shape[-2], device=depth_map.device), torch.arange(depth_map.shape[-1], device=depth_map.device))
    X = ((torch.multiply(u, depth_map) - cx * depth_map) / f)
    Y = -((torch.multiply(v, depth_map) - cy * depth_map) / f)
    Z = depth_map
    return X.flatten(), Y.flatten(), Z.flatten()


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
                               [0, 0, 0, 1.0]])

    return (dimX, dimY, dimZ), camera2frustum


def depth_to_gridspace(distance_map, intrinsic_path=None, down_scale_factor=1):
    input_depth = pyexr.open(distance_map).get("R")[:, :, 0]

    intrinsic = get_intrinsic(intrinsic_path)
    focal_length = intrinsic[0][0]

    # distance map to depth map
    transform = FromDistanceToDepth(focal_length)
    depthmap = transform(input_depth)

    return depthmap_to_gridspace(depthmap, intrinsic_path, down_scale_factor)

def depthmap_to_gridspace(depthmap, intrinsic_path=None, down_scale_factor=1):
    device = depthmap.device
    intrinsic = get_intrinsic(intrinsic_path)
    focal_length, cx, cy = intrinsic[0][0], intrinsic[0][2], intrinsic[1][2]
    
    # depth to camera space
    bs = depthmap.shape[0] #batch_size
    X, Y, Z = depth_to_camera(depthmap, focal_length, cx, cy)

    # get camera space to grid space transform
    intrinsic_inv = torch.inverse(intrinsic)
    frustum = generate_frustum([320, 240], intrinsic_inv, 0.4, 6.0)
    dims, camera2frustum = generate_frustum_volume(frustum, 0.05 * down_scale_factor)
    camera2frustum = camera2frustum.to(device)

    # depth from camera to grid space
    coords = torch.stack([X, Y, Z, torch.ones_like(X)])
    depth_in_gridspace = (camera2frustum @ coords)[:3, :].transpose(1,0).reshape(bs, -1, 3)

    return depth_in_gridspace

def get_intrinsic(intrinsic_path=None):
    if intrinsic_path is None:
        intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")

    intrinsic_line_0, intrinsic_line_1 = intrinsic_path.read_text().splitlines()[:2]
    focal_length = float(intrinsic_line_0[2:].split(',')[0])
    cx = float(intrinsic_line_0[2:-2].split(',')[2].strip())
    cy = float(intrinsic_line_1[1:-2].split(',')[2].strip())
    intrinsic = torch.tensor([[focal_length, 0, cx, 0], [0, focal_length, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return intrinsic


if __name__ == "__main__":
    from pathlib import Path

    output_pt_cloud_path = str("/home/alex/Documents/ifnet_scenes-main/ifnet_scenes/data/visualizations/overfit/00000/new_depthmap_to_pc.obj")
    focal_length = intrinsics_matrix[0][0]

    depth_map_path = str("/home/alex/Documents/ifnet_scenes-main/ifnet_scenes/data/visualizations/overfit/00000/depth_map.exr")
    depth_map = pyexr.open(str(depth_map_path)).get("Z")[:, :, 0]
    depth_map = torch.from_numpy(depth_map)

    pointcloud = depthmap_to_gridspace(depth_map)
    pointcloud = pointcloud.reshape(-1,3).squeeze()

    visualize_point_list(pointcloud, output_pt_cloud_path)