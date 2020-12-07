from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np

from data_processing.volume_reader import read_df
from util.visualize import visualize_point_list


class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, num_points, splitsdir):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (240 if (splitsdir == 'overfit') and split == 'train' else 1)
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(self.dataset_path) / "processed" / self.splitsdir / item
        sample_input = torch.from_numpy(np.load(sample_folder / "depth_grid.npz")['grid']).float()
        # sample_input = torch.nn.functional.interpolate(sample_input.unsqueeze(0).unsqueeze(0), scale_factor=1).squeeze()
        sample_target = torch.from_numpy(read_df(str(sample_folder / "target.df"))).float()
        # sample_target = torch.nn.functional.interpolate(sample_target.unsqueeze(0).unsqueeze(0), scale_factor=1).squeeze()
        points = []
        occupancies = []
        grids = []

        for sigma in ['0.10', '0.01']:
            sample_points_occ_npz = np.load(sample_folder / f"occupancy_{sigma}.npz")
            boundary_sample_points = sample_points_occ_npz['points']
            boundary_sample_coords = sample_points_occ_npz['grid_coords']
            boundary_sample_occupancies = sample_points_occ_npz['occupancies']
            subsample_indices = np.random.randint(0, boundary_sample_points.shape[0], self.num_points)
            points.extend(boundary_sample_points[subsample_indices])
            grids.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        sample_points = torch.from_numpy(np.array(points, dtype=np.float32))  # * (1 - 16 / 64))
        sample_occupancies = torch.from_numpy(np.array(occupancies, dtype=np.float32))
        sample_grid = torch.from_numpy(np.array(grids, dtype=np.float32))

        return {
            'name': item,
            'grid': sample_grid,
            'points': sample_points,
            'input': sample_input.unsqueeze(0),
            'occupancies': sample_occupancies,
            'target': sample_target.unsqueeze(0)
        }


if __name__ == "__main__":
    dataset = ImplicitDataset("train", "data", 3000, "overfit")
    print(dataset[0])  # TODO: Write a test case with visualizations
