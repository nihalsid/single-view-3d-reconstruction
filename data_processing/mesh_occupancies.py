from pathlib import Path
import trimesh
import numpy as np
import data_processing.implicit_waterproofing as iw
from util.visualize import visualize_point_list


def sample_points(mesh_path, dims, sample_num, sigma):
    mesh = trimesh.load(mesh_path)
    total_size = np.array(dims).max()
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


if __name__ == "__main__":
    dims = (139, 104, 112)

    mesh_path = Path("data") / "visualizations" / "overfit" / "00000" / "mesh.obj"
    output_occ_path = Path("data") / "visualizations" / "overfit" / "00000" / "occupied.obj"
    output_notocc_path = Path("data") / "visualizations" / "overfit" / "00000" / "not_occupied.obj"
    boundary_points, occupancies, grid_coords = sample_points(mesh_path, dims, 100000, 0.01)
    visualize_point_list(boundary_points[occupancies == 1, :], output_occ_path)
    visualize_point_list(boundary_points[occupancies == 0, :], output_notocc_path)
