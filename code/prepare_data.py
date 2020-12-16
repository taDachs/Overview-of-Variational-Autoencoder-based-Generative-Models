#!/usr/bin/env python3

import os
import argparse

from PIL import Image


def crop_dataset(source_dir, destination_dir):
    for img_path in os.listdir(source_dir):
        path = os.path.join(source_dir, img_path)
        img = Image.open(path)
        offset = 15
        cropped_img = img.crop((15 + offset, 35 + offset, 163 - offset, 183 - offset))
        save_path = os.path.join(destination_dir, img_path)
        cropped_img.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for preprocessing of the celeba dataset')
    parser.add_argument('--data', metavar='[DATASET PATH]', type=str, nargs=1, help='path to dataset')
    parser.add_argument('--dst', metavar='[DESTINATION PATH]', type=str, nargs=1, help='path to destination')

    args = parser.parse_args()

    crop_dataset(args.data, args.dst)
