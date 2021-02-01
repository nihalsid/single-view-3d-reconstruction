import numpy as np
from pathlib import Path
from data_processing.distance_to_depth import depth_to_gridspace
from data_processing.mesh_occupancies import sample_points
from data_processing.volume_reader import read_df
from util.visualize import visualize_sdf
from shutil import copyfile, move
import os

def process_sample(dataset_path, splitsdir, sample_name, down_scale_factor=1):
    # convert depth to grid
    dims = (round(139 / down_scale_factor), round(104 / down_scale_factor), round(112 / down_scale_factor))

    sample = Path(dataset_path) / "raw" / splitsdir / sample_name
    out = Path(dataset_path) / "processed" / splitsdir / sample_name
    out.mkdir(exist_ok=True, parents=True)

    depth_grid_space = depth_to_gridspace(str(sample / "distance.exr"), sample / "intrinsic.txt", down_scale_factor)
    grid = np.zeros(dims)
    to_int = lambda x: np.round(x.numpy()).astype(np.int32)
    grid[to_int(depth_grid_space[:, 0]), to_int(depth_grid_space[:, 1]), to_int(depth_grid_space[:, 2])] = 1
    np.savez_compressed(out / "depth_grid", grid=grid)

    df = read_df(str(sample / "distance_field.df"), down_scale_factor)
    visualize_sdf(df, sample / "mesh.obj", level=1.0)
    copyfile(str(sample / "distance_field.df"), out / "target.df")

    for sigma in [0.01, 0.1]:
        boundary_points, occupancies, grid_coords = sample_points(sample / "mesh.obj", dims, 100000, sigma)
        np.savez(out / f"occupancy_{sigma:.02f}", points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)

def process_sample_pipeline(dataset_path, splitsdir, down_scale_factor=1):
    # convert depth to grid
    dims = (round(139 / down_scale_factor), round(104 / down_scale_factor), round(112 / down_scale_factor))

    d_path = Path(dataset_path) / splitsdir
    scenes = sorted(os.listdir(d_path))
    for scene in scenes:
        views = sorted(os.listdir(d_path / scene))
        for view in views:
            sample = d_path / scene / view
            
            try:
                out = sample
                out.mkdir(exist_ok=True, parents=True)
                print("meshing file:", str(sample / "distance_field.df"))
                
                depth_grid_space = depth_to_gridspace(str(sample / "distance.exr"), Path(dataset_path) / "intrinsics.txt", down_scale_factor)
                
                grid = np.zeros(dims)
                to_int = lambda x: np.round(x.numpy()).astype(np.int32)
                grid[to_int(depth_grid_space[:, 0]), to_int(depth_grid_space[:, 1]), to_int(depth_grid_space[:, 2])] = 1
                
                np.savez_compressed(out / "depth_grid", grid=grid)
                df = read_df(str(sample / "distance_field.df"), down_scale_factor)
                visualize_sdf(df, sample / "mesh.obj", level=1.0)
                #copyfile(str(sample / "distance_field.df"), out / "target.df")

                for sigma in [0.01, 0.1]:
                    boundary_points, occupancies, grid_coords = sample_points(sample / "mesh.obj", dims, 100000, sigma)
                    np.savez(out / f"occupancy_{sigma:.02f}", points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
            
            # This catches meshes that are not of size dims (139, 104,112) and moves the view into another folder (quarantine)
            except IndexError:
                quarantine = Path(dataset_path) / "quarantine" / splitsdir / scene / view
                print("An Index-Error exception occurred, moving file: ", sample, "to ", quarantine) 
                move(sample, quarantine)
            # This catches empty meshes that can't be sampled  and moves the view into another folder (quarantine)
            except AttributeError:
                quarantine = Path(dataset_path) / "quarantine" / splitsdir / scene / view
                print("An Attribute-Error exception occurred, moving file: ", sample, "to ", quarantine) 
                move(sample, quarantine)

if __name__ == "__main__":
    path = "/media/alex/01D6C1999581FF10/Users/alexs/OneDrive/Desktop/3dfront_share/processed"
    process_sample_pipeline(path, "train", down_scale_factor=1)
