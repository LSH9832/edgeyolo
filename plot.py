import argparse
import os.path as osp

from edgeyolo.utils.plot_fig import plot, plot_ap, plot_all

import matplotlib
matplotlib.use('TkAgg')

DEFAULT_SIZE = (20, 9)
DEFAULT_SUFFIX = ["svg"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ap", "--ap", action="store_true")
    parser.add_argument("-loss", "--loss", action="store_true")
    parser.add_argument("-lr", "--lr", action="store_true")
    parser.add_argument("-a", "--all", action="store_true")

    parser.add_argument("-f", "--file", type=str, required=True)

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("--format", type=str, default=DEFAULT_SUFFIX, nargs="+")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.ap:
        plot_ap(args.file, show=not args.no_show, save=args.save, suffix=args.format, figsize=DEFAULT_SIZE)
    elif args.loss:
        plot(args.file, plot_type="loss", show=not args.no_show, save=args.save, suffix=args.format, figsize=DEFAULT_SIZE)
    elif args.lr:
        plot(args.file, plot_type="lr", show=not args.no_show, save=args.save, suffix=args.format, figsize=DEFAULT_SIZE)
    elif args.all:
        if osp.isfile(args.file):
            args.file = osp.dirname(args.file)
        plot_all(args.file, show=not args.no_show, figsize=DEFAULT_SIZE, save=args.save, suffix=args.format)
