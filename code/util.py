#!/usr/bin/env python3
import os
import json

import tensorflow.keras as keras
import tensorflow_probability as tfp
tfpl = tfp.layers

from plots import ProbabilisticAE, DeterministicAE


def substitute(s, substitution_dict):
    s = substitution_dict[s]
    if 'BETA' in s:
        s.replace('BETA', '')
        s = rf'$\beta${s}'
    return s


def wrap_model(encoder, decoder):
    if type(encoder.layers[-1]) == tfpl.IndependentNormal:
        ModelType = ProbabilisticAE
        latent_dims = encoder.layers[-1].output[1].shape[1]
    else:
        ModelType = DeterministicAE
        latent_dims = encoder.layers[-1].output.shape[1]

    return ModelType(encoder, decoder, latent_dims)


def load_encoder_decoder(path):
    encoder = keras.models.load_model(os.path.join(path, 'encoder'))
    decoder = keras.models.load_model(os.path.join(path, 'decoder'))

    return encoder, decoder


def parse_plot_config(config_path):
    with open(config_path) as json_file:
        data = json.load(json_file)

    models_dir = data.pop('model_path')
    substitution_dict = data.pop('substitute')

    model_names = []
    models = []
    feature_dict = {}
    range_dict = {}
    for model_name, params in data.items():
        if 'range' in params:
            range_dict[model_name] = params.pop('range')

        for feature, dim in params.items():
            if feature not in feature_dict:
                feature_dict[feature] = []

            feature_dict[feature].append(dim)

        encoder, decoder = load_encoder_decoder(os.path.join(models_dir, model_name))
        models.append(wrap_model(encoder, decoder))
        model_names.append(model_name)

    return models, model_names, feature_dict, range_dict, substitution_dict
