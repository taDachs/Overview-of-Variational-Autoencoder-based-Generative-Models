#!/usr/bin/env python3
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for training the models')
    parser.add_argument('--model', metavar='M', type=str, nargs=1, help='type of model')
    parser.add_argument('--data', metavar='[DATASET PATH]', type=str, nargs=1, help='path to dataset')
    parser.add_argument('--dst', metavar='[DESTINATION PATH]', type=str, nargs=1, help='path to model save destination')
    parser.add_argument('--epochs', metavar='N', type=int, default=30, nargs=1, help='Number of epochs trained')
    parser.add_argument('--learning-rate', metavar='l', type=float, default=0.001, nargs=1,
                        help='learning rate for model')
    parser.add_argument('--latent', metavar='L', type=int, default=32, nargs=1, help='dimension of latent space')
    parser.add_argument('--beta', metavar='B', type=float, default=1, nargs=1, help='beta regularizer')
    parser.add_argument('--threads', metavar='T', type=int, nargs=1, default=1,
                        help='number of worker threads for training process')
    parser.add_argument('--allow-growth', action='store_true',
                        help='if set, allows memory growth for gpu accelerated learning')

    args = parser.parse_args()

    model_type = args.model
    data_path = args.data
    save_path = args.dst
    epochs = args.epochs
    learning_rate = args.learning_rate
    latent_dims = args.latent
    beta = args.beta
    workers = args.threads
    allow_growth = args.growth

    if allow_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    img_width, img_height = (64, 64)
    input_shape = (img_width, img_height, 3)
    batch_size = 128

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        data_path,
        batch_size=batch_size,
        target_size=(img_width, img_height),
        class_mode=None
    )

    if model_type == 'ae':
        model = AE(latent_dims=latent_dims)
    elif model_type == 'vae':
        model = VAE(beta=1, tc=False, latent_dims=latent_dims)
    elif model_type == 'bvae':
        model = VAE(beta=beta, tc=False, latent_dims=latent_dims)
    elif model_type == 'btcvae':
        model = VAE(beta=beta, tc=True, latent_dims=latent_dims)
    else:
        raise NotImplementedError

    model.compile(optimizer=keras.optimizers.Adam(learning_rate))
    history = model.fit(train_generator, epochs=epochs, workers=workers)
    model.save(save_path)
