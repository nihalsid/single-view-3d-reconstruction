from pathlib import Path
import trimesh
import numpy as np
import data_processing.implicit_waterproofing as iw
from util.visualize import visualize_point_list
import torch


def sample_points(mesh_path, dims, sample_num, sigma):
    mesh = trimesh.load(mesh_path)
    total_size = np.array(dims)
    mesh.apply_translation(-np.array(dims)/2)
    mesh.apply_scale(1 / total_size)
    points = mesh.sample(sample_num)
    boundary_points = points + sigma * np.random.randn(sample_num, 3)
    random_points = np.random.uniform(-0.5, 0.5, size=(int(sample_num * 0.1), 3))
    boundary_points = np.vstack((boundary_points, random_points))
    grid_coords = boundary_points.copy()
    grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
    grid_coords = 2 * grid_coords
    occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]
    return boundary_points, occupancies, grid_coords

def determine_occupancy(mesh_path, points, dims=(139, 104, 112)):
    #normalize scale of points and mesh
    points[:,:, 0] -= (dims[0] / 2)
    points[:,:, 1] -= (dims[1] / 2)
    points[:,:, 2] -= (dims[2] / 2)
    points[:,:, 0] = (dims[0])
    points[:,:, 1] = (dims[1])
    points[:,:, 2] = (dims[2])

    #please vectorize
    occs = []
    for i, path in enumerate(mesh_path): 

        mesh = trimesh.load(path)
        size = np.array(dims)
        mesh.apply_translation(-size/2)
        mesh.apply_scale(1 / size)

        #points are in z,y,x system and mesh in x,y,z. Swap x & z. (Not sure about scale)
        #grid_coords = points[i].clone().detach()
        #grid_coords[:, 0], grid_coords[:, 2] = points[i,:, 2].squeeze(), points[i,:, 0].squeeze()
        #grid_coords = 2 * grid_coords

        #determine occupancy of points
        occupancies = iw.implicit_waterproofing(mesh, points[i])[0]
        occs.extend(occupancies)
    #return points, occupancies, grid_coords
    occs = torch.from_numpy(np.array(occs))
    occs = occs.to(torch.float)
    return points, occs


if __name__ == "__main__":
    from data_processing.distance_to_depth import depth_to_gridspace
    dims = (139, 104, 112)

    dinstace_field_path = Path("data") / "raw" / "overfit" / "00000" / "distance.exr"

    pointcloud_occ = depth_to_gridspace(str(dinstace_field_path))
    pointcloud_occ = pointcloud_occ.reshape(1,-1,3).numpy()

    pointcloud_notocc = depth_to_gridspace(str(dinstace_field_path), eps = 0.1)
    pointcloud_notocc = pointcloud_notocc.reshape(1,-1,3).numpy()

    mesh_path = Path("data") / "visualizations" / "overfit" / "00000" / "mesh.obj"
    output_occ_path = Path("data") / "test"  / "pc_occupied.obj"
    output_notocc_path = Path("data") / "test" / "pc_not_occupied.obj"

    boundary_points, occupancies = determine_occupancy([str(mesh_path)], pointcloud_occ, dims)
    visualize_point_list(boundary_points[0,occupancies == 1, :], output_occ_path)
    visualize_point_list(boundary_points[0,occupancies == 0, :], output_notocc_path)
