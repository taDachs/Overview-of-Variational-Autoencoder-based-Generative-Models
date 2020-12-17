#!/usr/bin/env sh

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 [DATASET_DIR]" >&2
  exit 1
fi

DATASET_DIR=$(readlink -f "$1")

python3 code/train_model.py --data "$DATASET_DIR" --model ae --epochs 1 --learning-rate 0.0001 --latent 32 --dst ./models/ae --allow-growth
python3 code/train_model.py --data "$DATASET_DIR" --model vae --epochs 1 --learning-rate 0.0001 --latent 32 --dst ./models/vae --allow-growth
python3 code/train_model.py --data "$DATASET_DIR" --model bvae --epochs 1 --learning-rate 0.0001 --latent 32 --dst ./models/bvae --allow-growth
python3 code/train_model.py --data "$DATASET_DIR" --model btcvae --epochs 1 --learning-rate 0.0001 --latent 32 --dst ./models/btcvae --allow-growth
