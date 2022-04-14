import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z from a normal distribution.
    Useful for variational autoencoder
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Object with custom classes. Important to be able and load saved models that utilize them
custom_objects = {'Sampling': Sampling}