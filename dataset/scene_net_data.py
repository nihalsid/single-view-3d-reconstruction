from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import pyexr
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import torchvision.transforms.functional as F

from data_processing.distance_to_depth import FromDistanceToDepth
from data_processing.volume_reader import read_df

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class scene_net_data(Dataset):

    def __init__(self, split, dataset_path, num_points, splitsdir, kwargs=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (500 if (splitsdir == 'overfit') and split == 'train' else 1)
        self.num_points = num_points
        self.input_transform = Compose(
            [
                # SquarePad(),
                # Resize((kwargs.W, kwargs.W)), #check whether needed
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.target_transform = ToTensor()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(self.dataset_path) / "raw" / self.splitsdir / item
        df_foler = Path(self.dataset_path) / "processed" / self.splitsdir / item
        
        image = Image.open(sample_folder / "rgb.png")
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rgb_img = self.input_transform(image)

        sample_target = torch.from_numpy(read_df(str(df_foler / "target.df"))).float().unsqueeze(0)

        points = []
        occupancies = []
        grids = []

        for sigma in ['0.10', '0.01']:
            sample_points_occ_npz = np.load(df_foler / f"occupancy_{sigma}.npz")
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
        sample_input = torch.from_numpy(np.load(df_foler / "depth_grid.npz")['grid']).float()

        distance_map = pyexr.open(str(sample_folder / "distance.exr")).get("R")[:, :, 0]

        #depthmap target
        intrinsic_line_0 = (sample_folder / "intrinsic.txt").read_text().splitlines()[0]
        focal_length = float(intrinsic_line_0[2:].split(',')[0])
        transform = FromDistanceToDepth(focal_length)
        depth_map = transform(distance_map).numpy().astype('float32', casting='same_kind')
        depth_map = np.flip(depth_map, 1)
        depth_flipped = depth_map.copy()
        depthmap_target = self.target_transform(depth_flipped)    
        
        return {
            'name': item,
            'rgb': rgb_img,
            'grid': sample_grid,
            'points': sample_points,
            'input': sample_input.unsqueeze(0),
            'occupancies': sample_occupancies,
            'target': sample_target.unsqueeze(0),
            'depthmap_target': depthmap_target
        }