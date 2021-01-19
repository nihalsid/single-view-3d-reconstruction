import argparse
import glob
from pathlib import Path
import shutil
import os
import math

# Create split files given percentages for each subset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create split files'
    )

    parser.add_argument('--samples_count', type=int, default=21264)
    parser.add_argument('--train_percentage', type=float, default=0.70)
    parser.add_argument('--val_percentage', type=float, default=0.20)
    parser.add_argument('--test_percentage', type=float, default=0.10)
    parser.add_argument('--target_path', type=str, default="../data/splits/full/")

    args = parser.parse_args()

    samples_count = args.samples_count
    train_percentage = args.train_percentage
    val_percentage = args.val_percentage
    test_percentage = args.test_percentage
    target_path = Path(args.target_path)

    if not os.path.exists(str(target_path)):
        os.makedirs(str(target_path))

    train_start = 0
    train_end = math.floor(samples_count * train_percentage)

    val_start = train_end + 1
    val_end = train_end + math.floor(samples_count * val_percentage)

    test_start = val_end + 1
    test_end = val_end + math.ceil(samples_count * test_percentage)

    train = []
    for i in range(train_start, train_end + 1):
        train.append('{:0>4}'.format(i))

    val = []
    for i in range(val_start, val_end + 1):
        val.append('{:0>4}'.format(i))

    test = []
    for i in range(test_start, test_end + 1):
        test.append('{:0>4}'.format(i))

    train_file = open(str(target_path / "train.txt"), 'w')
    train_file.write('\n'.join(train))
    train_file.close()

    val_file = open(str(target_path / "val.txt"), 'w')
    val_file.write('\n'.join(val))
    val_file.close()

    test_file = open(str(target_path / "test.txt"), 'w')
    test_file.write('\n'.join(test))
    test_file.close()
