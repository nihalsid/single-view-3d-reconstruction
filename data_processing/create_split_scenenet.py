import argparse
import glob
from pathlib import Path
import shutil
import os
import math
import numpy as np
import random

# Create split files given percentages for each subset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create split files'
    )

    parser.add_argument('--samples_count', type=int, default=2735) #scenes: 2735train + 592val
    parser.add_argument('--train_percentage', type=float, default=0.35)
    parser.add_argument('--val_percentage', type=float, default=0.10)
    parser.add_argument('--test_percentage', type=float, default=0.05)
    parser.add_argument('--target_path', type=str, default="data/splits/test_split/")
    #parser.add_argument('--rnd', type=str, default="data/splits/full_data/")
    parser.add_argument('--subsample', type=float, default=0.05)

    parser.add_argument('--dataset_path', type=str, default="/media/alex/01D6C1999581FF10/Users/alexs/OneDrive/Desktop/3dfront_share/processed/")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    samples_count = args.samples_count
    train_percentage = args.train_percentage
    val_percentage = args.val_percentage
    test_percentage = args.test_percentage
    target_path = Path(args.target_path)
    subsample = args.subsample

    splitsdir = {'train':[], 'val':[], 'test':[]}
    #splitpercent = {'train': train_percentage, 'myval': val_percentage, 'mytest': test_percentage}
    
    for split in splitsdir.keys():
        d_path = Path(dataset_path) / split
        scenes = sorted(os.listdir(d_path))
        for scene in scenes:
            views = sorted(os.listdir(d_path / scene))
            for view in views:
                sample = d_path / scene / view
                splitsdir[split].append(str(sample))
            
    if not os.path.exists(str(target_path)):
        os.makedirs(str(target_path))
        
    # if you don't want to train on all
    subsamples = {'train': int(subsample*len(splitsdir['train'])), 'val': int(subsample*len(splitsdir['val'])), 'test': int(subsample*len(splitsdir['test']))}

    for split in splitsdir.keys():
        with open(str(target_path / "{}.txt".format(split)), 'w') as split_file:
            if split == 'train':
                random.shuffle(splitsdir[split])
            splitsdir[split] = '\n'.join(splitsdir[split][:subsamples[split]])
            split_file.writelines(splitsdir[split])
