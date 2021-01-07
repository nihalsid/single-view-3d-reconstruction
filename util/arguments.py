import argparse
from datetime import datetime
from pathlib import Path
from random import randint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=0, help='gpus')
    parser.add_argument('--sanity_steps', type=int, default=0, help='overfit multiplier')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--splitsdir', type=str, default='overfit', help='resume checkpoint')
    parser.add_argument('--datasetdir', type=str, help='datasetdir', default='data')
    parser.add_argument('--val_check_percent', type=float, default=1.0, help='percentage of val checked')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='check val every fraction of epoch')
    parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
    parser.add_argument('--lr', type=float, default=0.000075, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--experiment', type=str, default='fast_dev', help='experiment directory')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument("--W", type=int, default=256)

    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--net_res', type=int, default=128, help='Architecture of the Network and number of features')
    parser.add_argument('--inf_res', type=int, default=1, help='Multiple of inference resolution per training grid resolution')

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = randint(0, 999)

    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)

    args.experiment = f"{datetime.now().strftime('%d%m%H%M')}_{args.experiment}"
    if args.resume is not None and not args.new_exp_for_resume:
        args.experiment = Path(args.resume).parents[0].name

    return args
