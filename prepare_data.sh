#!/usr/bin/env sh

if [ "$#" -ne 2 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 [DATASET_DIR] [DESTINATION_DIR]" >&2
  exit 1
fi

DATASET_DIR=$(readlink -f $1)
DESTINATION_DIR="$2"

mkdir -p "$DESTINATION_DIR"
DESTINATION_DIR=$(readlink -f $2)

python3 code/prepare_data.py --data "$DATASET_DIR" --dst "$DESTINATION_DIR"
