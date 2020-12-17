#!/usr/bin/env sh

if [ "$#" -ne 2 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 [DATASET_DIR] [CONFIG PATH]" >&2
  exit 1
fi

DATASET_DIR=$(readlink -f $1)
CONFIG_PATH=$(readlink -f $2)

python3 code/generate_plots.py --data "$DATASET_DIR" --num_images 5 --config "$CONFIG_PATH"
