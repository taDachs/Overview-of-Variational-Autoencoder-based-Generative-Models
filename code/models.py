import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_probability as tfp

tfpl = tfp.layers
tfd = tfp.distributions

img_width, img_height = (64, 64)
input_shape = (img_width, img_height, 3)

def get_encoder(latent_dim, probabilistic=False):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 4, strides=2, padding="same", use_bias=False)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(latent_dim, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(tfpl.IndependentNormal.params_size(latent_dim) / 2)(x)
    if probabilistic:
        sigma = layers.Dense(tfpl.IndependentNormal.params_size(latent_dim) / 2)(x)
        sigma = tf.exp(sigma)
        x = tf.concat((mu, sigma), axis=1)
        x = tfpl.IndependentNormal(latent_dim, validate_args=True)(x)
    else:
        x = mu
    encoder = keras.Model(encoder_inputs, x, name='encoder')
    return encoder


def get_decoder(latent_dim, probabilistic=False):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Reshape((1, 1, latent_dim))(latent_inputs)
    x = layers.Conv2DTranspose(512, 1, strides=1, padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=1, padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    mu = layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same', use_bias=False, activation=None)(x)
    if probabilistic:
        mu = layers.Flatten()(mu)
        mu = keras.activations.sigmoid(mu)
        # mu = layers.Dense(tfpl.IndependentNormal.params_size(input_shape) / 2)(mu)
        sigma = tf.fill(tf.shape(mu), 0.01)
        x = tf.concat((mu, sigma), axis=1)
        # x = tfkl.Dense(tfpl.IndependentNormal.params_size(input_shape))(x)
        x = tfpl.IndependentNormal(input_shape, validate_args=True)(x)
    else:
        x = mu
    decoder = keras.Model(latent_inputs, x, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, beta=1, tc=False, latent_dim=32, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)
        self.latent_dim = latent_dim
        self.encoder = get_encoder(self.latent_dim, probabilistic=True)
        self.decoder = get_decoder(self.latent_dim, probabilistic=True)
        self.beta = beta
        self.tc = tc

        self.set_size = 202240
        self.batch_size = 128

    def call(self, input):
        return None

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            qz_x = self.encoder(data)
            z = qz_x.sample()
            px_z = self.decoder(z)
            x = px_z.mean()

            log_px_z = px_z.log_prob(data)
            log_qx_z = qz_x.log_prob(z)
            log_pz = self.prior.log_prob(z)
            kl = qz_x.log_prob(z) - self.prior.log_prob(z)
            tc = self.total_correlation(z, qz_x)

            if self.tc:
                elbo = log_px_z - kl - (self.beta - 1) * tc
            else:
                elbo = log_px_z - self.beta * kl

            total_loss = -elbo

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'loss': total_loss,
            'logpx': log_px_z,
            'kl': kl,
            'tc': tc
        }

    def total_correlation(self, zs, qz_x):
        mu = qz_x.mean()
        sigma = qz_x.variance()

        log_qz_prob = tfd.Normal(tf.expand_dims(mu, 0), tf.expand_dims(sigma, 0)).log_prob(tf.expand_dims(zs, 1))

        log_qz_prod = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1, keepdims=False)

        log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2, keepdims=False), axis=1, keepdims=False)

        return log_qz - log_qz_prod

    def save(self, epochs_trained, learning_rate):
        self.encoder.save(f'vae_{self.beta}_{self.latent_dim}_{epochs_trained}_{int(1 / learning_rate)}/encoder')
        self.decoder.save(f'vae_{self.beta}_{self.latent_dim}_{epochs_trained}_{int(1 / learning_rate)}/decoder')


class AE(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = get_encoder(self.latent_dim)
        self.decoder = get_decoder(self.latent_dim)

    def call(self, input):
        return None

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            x = self.decoder(z)

            mse = tf.reduce_sum(tf.keras.losses.mse(data, x), axis=[1, 2])

        grads = tape.gradient(mse, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'loss': mse
        }

    def save(self, epochs_trained, learning_rate):
        self.encoder.save(f'ae_{self.latent_dim}_{epochs_trained}_{int(1 / learning_rate)}/encoder')
        self.decoder.save(f'ae_{self.latent_dim}_{epochs_trained}_{int(1 / learning_rate)}/decoder')