import argparse
import glob
from pathlib import Path
import shutil
import os

def check_sample_zipped_correctly(sample):
    zipped_correctly = True

    path = sample[0].split('/')[-2]
    sample_number = sample[0].split('/')[-1].split('_')[1].split('.')[0]
    for i in range(1, len(sample)):
        zipped_correctly &= path == sample[i].split('/')[-2]
        path = sample[i].split('/')[-2]

        zipped_correctly &= sample_number == sample[i].split('/')[-1].split('_')[1].split('.')[0]
        sample_number = sample[i].split('/')[-1].split('_')[1].split('.')[0]

    return zipped_correctly


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split Data'
    )

    parser.add_argument('--source_path', type=str)
    parser.add_argument('--target_path', type=str, default="../data/raw/full/")
    
    args = parser.parse_args()

    source_path = Path(args.source_path)
    target_path = Path(args.target_path)

    if not os.path.exists(str(target_path)):
        os.makedirs(str(target_path))

    scenes_folders = glob.glob(str(source_path / '2d' / "**"))
    scenes_folders = sorted([folder.split("/")[-1] for folder in scenes_folders])

    print(len(scenes_folders))
    # Loop over 2d and 3d files and copy them to destination
    scene_counter = 0
    for scene_folder in scenes_folders:
        views = zip(sorted(glob.glob(str(source_path / '2d' / scene_folder / "*.png"))),
        sorted(glob.glob(str(source_path / '2d' / scene_folder / "campose*.npy"))),
        sorted(glob.glob(str(source_path / '2d' / scene_folder / "distance*.exr"))),
        sorted(glob.glob(str(source_path / '3d' / scene_folder / "distance_field*.df")))
        )
        for i, sample in enumerate(views):
            sample_path = target_path / ('{:0>4}'.format(scene_counter)) / str(i)
            os.makedirs(str(sample_path))

            shutil.copy(sample[0], str(sample_path / "rgb.png"))
            shutil.copy(sample[1], str(sample_path / "campose.npy"))
            shutil.copy(sample[2], str(sample_path / "distance.exr"))
            shutil.copy(sample[3], str(sample_path / "distance_field.df"))
        scene_counter += 1
    

