from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import pyexr
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import torchvision.transforms.functional as F

from data_processing.distance_to_depth import FromDistanceToDepth

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class ScenesDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, kwargs=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (500 if (splitsdir == 'overfit') and split == 'train' else 1)
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
        intrinsic_line_0 = (sample_folder / "intrinsic.txt").read_text().splitlines()[0]
        focal_length = float(intrinsic_line_0[2:].split(',')[0])
        
        image = Image.open(sample_folder / "rgb.png")
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        sample_input = self.input_transform(image)
        # All distance channels are of the same value
        distance_map = pyexr.open(str(sample_folder / "distance.exr")).get("R")[:, :, 0]

        # distance map to depth map
        transform = FromDistanceToDepth(focal_length)
        depth_map = transform(distance_map).astype('float32', casting='same_kind')
        depth_map = np.flip(depth_map, 1)
        depth_flipped = depth_map.copy()
        
        sample_target = self.target_transform(depth_flipped)

        return {
            'name': item,
            'input': sample_input,
            'target': sample_target
        }