#!/usr/bin/env sh

DATASET_DIR=$1
MODEL_DIR=$2
LATENT_FEATURE_CONFIG=$3

python3 code/generate_plots.py --data "$DATASET_DIR" --models "$MODEL_DIR" --num_images 5 --latent_feature "$LATENT_FEATURE_CONFIG"