#!/usr/bin/env python3
import numpy as np

import tensorflow_probability as tfp

tfpl = tfp.layers


def wrap_model(encoder, decoder):
    if type(encoder.layers[-1]) == tfpl.IndependentNormal:
        model_wrapper = ProbabilisticAE
        latent_dims = encoder.layers[-1].output[1].shape[1]
    else:
        model_wrapper = DeterministicAE
        latent_dims = encoder.layers[-1].output.shape[1]

    return model_wrapper(encoder, decoder, latent_dims)


def clip(x):
    return np.clip(x, 0, 1)


class Model:
    def __init__(self, encoder, decoder, latent_dim, output_shape=(64, 64, 3)):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.output_shape = output_shape


class DeterministicAE(Model):
    def __init__(self, encoder, decoder, latent_dim, output_shape=(64, 64, 3)):
        super().__init__(encoder, decoder, latent_dim, output_shape)

    def get_latent(self, img):
        return self.encoder(np.expand_dims(img, 0))[0]

    def get_reconstruction(self, z):
        return clip(self.decoder(np.expand_dims(z, 0))[0])

    def sample(self):
        z = np.random.normal(size=(1, self.latent_dim))
        return clip(self.decoder(z)[0])


class ProbabilisticAE(Model):
    def __init__(self, encoder, decoder, latent_dim, output_shape=(64, 64, 3)):
        super().__init__(encoder, decoder, latent_dim, output_shape)

    def get_latent(self, img):
        qz_n = self.encoder(np.expand_dims(img, 0))
        return qz_n.sample()[0]

    def get_reconstruction(self, z):
        px_z = self.decoder(np.expand_dims(z, 0))
        return clip(px_z.mean()[0])

    def sample(self):
        z = np.random.normal(size=(1, self.latent_dim))
        px_z = self.decoder(z)

        return clip(px_z.mean()[0])
