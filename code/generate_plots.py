#!/usr/bin/env python3
import argparse
import random
import os

import numpy as np

from PIL import Image

from plots import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for preprocessing of the celeba dataset')
    parser.add_argument('--data', metavar='[DATASET PATH]', type=str, nargs=1, help='path to dataset')
    parser.add_argument('--dst', metavar='[DESTINATION PATH]', type=str, nargs=1, help='path to plot destination')
    parser.add_argument('--config', metavar='[CONFIG PATH]', type=str, nargs=1, help='path to config')
    parser.add_argument('--num_images', metavar='N', type=int, nargs=1, default=5,
                        help='path to folder containing models')

    args = parser.parse_args()

    data_path = args.data
    num_images = args.num_images
    models_path = args.models
    config_path = args.config
    plot_path = args.dst
    imgs = []

    path_list = os.listdir(data_path)
    random.shuffle(path_list)

    for img_path in path_list:
        img = Image.open(os.path.join(data_path, img_path))
        img.thumbnail((64, 64), Image.ANTIALIAS)
        img = np.array(img, dtype=float)
        img *= 1.0 / 255.0

        imgs.append(img)

        if len(imgs) >= num_images:
            break

    models, model_names, feature_dict, range_dict, substitution_dict = parse_plot_config(config_path)

    fig, ax = plot_comparison_disentanglement(imgs=imgs, models=models, model_names=model_names,
                                              feature_dict=feature_dict, range_dict=range_dict,
                                              substitution_dict=substitution_dict)

    fig.savefig(os.path.join(plot_path, 'disentanglement_comparison.pgf'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(plot_path, 'disentanglement_comparison.pdf'), dpi=200, bbox_inches='tight')

    fig, ax = plot_comparison_generative(models=models, model_names=model_names, size=(num_images, num_images),
                                         substitution_dict=substitution_dict)
    fig.savefig(os.path.join(plot_path, 'generative_comparison.pgf'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(plot_path, 'generative_comparison.pgf'), dpi=200, bbox_inches='tight')