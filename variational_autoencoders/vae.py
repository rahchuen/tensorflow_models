import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import functools
from functional import compose, partial
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.examples.tutorials.mnist import input_data


def compose_funcs(*args):
    """Multiple function composition
    i.e. composed = compose_funcs([f, g, h])
         composed(x) == f(g(h(x)))
    """
    return partial(functools.reduce, compose)(*args)


class VariationalAutoEncoder(object):
    def __init__(self,
                 architecture,
                 dropout,
                 activation,
                 initialization,
                 learning_rate):
        self.architecture = architecture
        self.dropout = dropout
        self.activation = activation
        self.initialization = initialization
        self.learning_rate = learning_rate

        # Build the graph
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, [None, 784])
        self._build_graph()

        self.tf_saver = tf.train.Saver()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.mu, self.logvar = self._encoder(self.input)
        self.z = self._reparametrize(self.mu, self.logvar)
        self.x_recons = self._decoder(self.z)
        self.loss = self.get_vae_loss(self.input, self.x_recons, self.mu, self.logvar)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _encoder(self, x):
        # probabilistic encoder: q(z|x)
        with tf.variable_scope("Encoder"):
            encoding = [Dense(units, activation=self.activation, kernel_initializer=self.initialization)
                        for units in reversed(self.architecture[1:-1])]
            x_encoded = compose_funcs(encoding)(x)
            z_mu = Dense(self.architecture[-1])(x_encoded)
            z_log_var = Dense(self.architecture[-1])(x_encoded)
            tf.summary.histogram('z_mu', z_mu)
            tf.summary.histogram('z_log_var', z_log_var)
        return z_mu, z_log_var

    def _decoder(self, z):
        # probabilistic decoder: p(x|z)
        with tf.variable_scope("Decoder"):
            decoding = [Dense(units, activation=activation, kernel_initializer=initialization)
                        for units in architecture[1:-1]]
            x_decoded = compose_funcs(decoding)(z)
            x_recons = tf.keras.layers.Dense(architecture[0], activation=tf.nn.sigmoid)(x_decoded)
        return x_recons

    @staticmethod
    def _reparametrize(mu, log_var):
        # z = z_mean + z_sigma * e, where e~N(0,1)
        with tf.variable_scope("Reparametrization Trick"):
            esp = tf.random.normal(tf.shape(log_var), mean=0, stddev=1.0)
            z = tf.add(mu, tf.mul(tf.exp(log_var / 2), esp))
            tf.summary.histogram('z_sampled', z)
        return z

    @staticmethod
    def calculate_cross_entropy_loss(x, x_hat, epsilon=1e-10):
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy_loss = x * tf.log(epsilon + x_hat) + (1 - x) * tf.log(epsilon + 1 - x_hat)
            cross_entropy_loss = -1 * tf.reduce_sum(cross_entropy_loss, 1)
        return cross_entropy_loss

    def get_vae_loss(self, x, x_recons, mu, log_var):
        # Consists of reconstruction loss + Latent loss
        with tf.name_scope('recons_loss'):
            recons_loss = self.calculate_cross_entropy_loss(x, x_recons)
            tf.summary.histogram('recons_loss', recons_loss)

        # Latent loss based on appendix B in https://arxiv.org/pdf/1312.6114.pdf
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        with tf.name_scope('latent_loss'):
            latent_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), 1)
            tf.summary.histogram('latent_loss', latent_loss)

        with tf.name_scope('vae_loss'):
            vae_loss = tf.reduce_mean(recons_loss + latent_loss)
            tf.summary.histogram('vae_loss', vae_loss)

        return vae_loss

    def train_one_step(self, X):
        # train one mini batch
        summary_writer = tf.summary.FileWriter('/home/rachel/vae/logs', self.sess.graph)
        merge = tf.summary.merge_all()
        summary, train_op, cost, z = self.sess.run(
            [merge, self.optimizer, self.loss, self.z],
            feed_dict={self.input: X})
        summary_writer.add_summary(summary)
        return cost, z

    def generate_sample(self, z):
        # Generate reconstructed x_hat from z. If z is not given,
        # z is sampled from prior z~N(0,1)
        if z is None:
            z = tf.np.random.normal(size=self.architecture[-1])
        return self.sess.run(self.x_recons, feed_dict={self.z: z})

    def reconstruct(self, X):
        # Reconstruct using given mini batch of data
        return self.sess.run(self.x_recons, feed_dict={self.input: X})

    def save(self, check_point_file ='vae_model.ckpt'):
        save_path = self.tf_saver.save(self.sess, check_point_file)
        print("saved weights to " + save_path)

    def load(self, check_point_file='vae_model.ckpt'):
        self.tf_saver.restore(self.sess, check_point_file)
        print("loaded weights from " + check_point_file)


def train(architecture, dropout, activation, initialization, learning_rate, num_epochs):
    vae = VariationalAutoEncoder(architecture, dropout, activation, initialization, learning_rate)
    mnist = input_data.read_data_sets('./data/MNIST_data')

    for epoch in range(num_epochs):
        total_batch = int(55000 / 1000)
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(1000)
            cost, z_code = vae.train_one_step(batch_xs)

        if epoch % 10 == 0:
            print('Epoch', epoch, 'cost', cost)
            samples = vae.generate_sample(z_code)
            plt.subplot(122)
            curr_img = samples[0].reshape(28, 28)
            plt.imshow(curr_img, cmap='gray')
            plt.show()

