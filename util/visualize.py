import marching_cubes as mc
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path
import pyexr
import torch


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s >= 0.5)], axis=1)


def visualize_point_list(grid, output_path):
    f = open(output_path, "w")
    for i in range(grid.shape[0]):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        c = [1, 1, 1]
        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def visualize_sdf(sdf, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes(sdf.astype(float), level)
    mc.export_obj(vertices, triangles, output_path)


def visualize_grid(grid, output_path):
    point_list = to_point_list(grid)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)
        
def visualize_depthmap(depthmap, output_path, flip=False):
    if isinstance(depthmap, np.ndarray):
        depthmap = depthmap.squeeze()

    elif isinstance(depthmap, torch.Tensor):
        depthmap = depthmap.squeeze().cpu().numpy()

    else:
        raise NotImplementedError

    if flip:
        depthmap = np.flip(depthmap, axis=1)
    rescaled = (255.0 / depthmap.max() * (depthmap - depthmap.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(str(output_path) +'.png')
    pyexr.write(str(output_path) +'.exr', depthmap)

def scale(path):
    dims = (139, 104, 112)
    mesh = trimesh.load(path, process=False)
    total_size = np.array(dims)
    #mesh.apply_translation(-np.array(dims)/2)
    mesh.apply_scale(1 / total_size)
    new_path = str(path)[:-4] + "_scaled.obj"
    print(new_path)
    mesh.export(new_path)

if __name__ == "__main__":
    path = Path("/home/alex/Documents/ifnet_scenes-main/ifnet_scenes/data/visualizations/overfit/00000")
    path = path / "mesh.obj"
    scale(path)