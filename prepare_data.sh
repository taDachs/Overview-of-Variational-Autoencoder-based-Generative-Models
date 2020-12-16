#!/usr/bin/env sh

DATASET_DIR=$1
DESTINATION_DIR=$1

python3 code/prepare_data.py --data "$DATASET_DIR" --dst "$DESTINATION_DIR"