#!/usr/bin/env sh

DATASET_DIR=$1
CONFIG_PATH=$2

python3 code/generate_plots.py --data "$DATASET_DIR" --num_images 5 --config "$CONFIG_PATH"