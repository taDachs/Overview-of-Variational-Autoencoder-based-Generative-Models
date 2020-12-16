#!/usr/bin/env sh

DATASET_DIR=$1

python3 code/train_model.py --data "$DATASET_DIR" --model ae --epochs 30 --learning_rate 0.0001 --latent 32 --output_path ./models/ae
python3 code/train_model.py --data "$DATASET_DIR" --model vae --epochs 30 --learning_rate 0.0001 --latent 32 --output_path ./models/vae
python3 code/train_model.py --data "$DATASET_DIR" --model bvae --epochs 30 --learning_rate 0.0001 --latent 32 --output_path ./models/bvae
python3 code/train_model.py --data "$DATASET_DIR" --model btcvae --epochs 30 --learning_rate 0.0001 --latent 32 --output_path ./models/btcvae
