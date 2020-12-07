import marching_cubes as mc
import numpy as np
import trimesh


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s == 1)], axis=1)


def visualize_point_list(grid, output_path):
    f = open(output_path, "w")
    for i in range(grid.shape[0]):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        c = [1, 1, 1]
        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def visualize_sdf(sdf, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes(sdf, level)
    mc.export_obj(vertices, triangles, output_path)


def visualize_grid(grid, output_path):
    point_list = to_point_list(grid)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)
