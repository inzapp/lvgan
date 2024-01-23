"""
Authors : inzapp

Github url : https://github.com/inzapp/lvgan

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, generate_shape, latent_dim):
        self.generate_shape = generate_shape
        self.latent_dim = latent_dim
        self.ae = None
        self.ae_e = None
        self.ae_d = None
        self.lvgan = None
        self.lvgan_g = None
        self.lvgan_d = None
        self.extra_strides = [64, 128, 256, 512]
        self.stride = self.calc_stride(self.generate_shape)
        self.latent_rows = generate_shape[0] // self.stride
        self.latent_cols = generate_shape[1] // self.stride

    def calc_stride(self, generate_shape):
        stride = 32
        min_size = min(generate_shape[:2])
        for v in self.extra_strides:
            if min_size >= v and min_size % v == 0:
                stride = v
            else:
                break
        return stride

    def build(self, lvgan_g=None, lvgan_d=None, ae_e=None, ae_d=None):
        assert self.generate_shape[0] % 32 == 0 and self.generate_shape[1] % 32 == 0
        if lvgan_g is None:
            lvgan_g_input, lvgan_g_output = self.build_lvgan_g(bn=True)
            self.lvgan_g = tf.keras.models.Model(lvgan_g_input, lvgan_g_output)
        else:
            lvgan_g_input, lvgan_g_output = lvgan_g.input, lvgan_g.output
            self.lvgan_g = lvgan_g

        if lvgan_d is None:
            lvgan_d_input, lvgan_d_output = self.build_lvgan_d(bn=False)
            self.lvgan_d = tf.keras.models.Model(lvgan_d_input, lvgan_d_output)
        else:
            lvgan_d_input, lvgan_d_output = lvgan_d.input, lvgan_d.output
            self.lvgan_d = lvgan_d

        if ae_e is None:
            ae_e_input, ae_e_output = self.build_ae_e(bn=True)
            self.ae_e = tf.keras.models.Model(ae_e_input, ae_e_output)
        else:
            ae_e_input, ae_e_output = ae_e.input, ae_e.output
            self.ae_e = ae_e

        if ae_d is None:
            ae_d_input, ae_d_output = self.build_ae_d(bn=True)
            self.ae_d = tf.keras.models.Model(ae_d_input, ae_d_output)
        else:
            ae_d_input, ae_d_output = ae_d.input, ae_d.output
            self.ae_d = ae_d

        lvgan_output = self.lvgan_d(lvgan_g_output)
        self.lvgan = tf.keras.models.Model(lvgan_g_input, lvgan_output)
        ae_output = self.ae_d(ae_e_output)
        self.ae = tf.keras.models.Model(ae_e_input, ae_output)
        return self.ae, self.ae_e, self.ae_d, self.lvgan, self.lvgan_g, self.lvgan_d

    def build_lvgan_g(self, bn):
        lvgan_g_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = lvgan_g_input
        x = self.dense(x, 512, activation='leaky', bn=bn)
        x = self.dense(x, 512, activation='leaky', bn=bn)
        lvgan_g_output = self.batch_normalization(self.dense(x, self.latent_dim, activation='linear'))
        return lvgan_g_input, lvgan_g_output

    def build_lvgan_d(self, bn):
        lvgan_d_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = lvgan_d_input
        x = self.dense(x, 512, activation='leaky', bn=bn)
        x = self.dense(x, 512, activation='leaky', bn=bn)
        lvgan_d_output = self.dense(x, 1, activation='linear')
        return lvgan_d_input, lvgan_d_output

    def build_ae_e(self, bn):
        ae_e_input = tf.keras.layers.Input(shape=self.generate_shape)
        x = ae_e_input
        x = self.conv2d(x, 32, 5, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 64, 5, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 128, 5, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 256, 5, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 256, 5, 2, activation='leaky', bn=bn)
        for extra_stride in self.extra_strides:
            if self.stride >= extra_stride:
                x = self.conv2d(x, 256, 5, 2, activation='leaky', bn=bn)
            else:
                break
        x = self.flatten(x)
        ae_e_output = self.dense(x, self.latent_dim, activation='linear')
        return ae_e_input, ae_e_output

    def build_ae_d(self, bn):
        ae_d_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = ae_d_input
        x = self.dense(x, self.latent_rows * self.latent_cols * 256, activation='leaky', bn=bn)
        x = self.reshape(x, (self.latent_rows, self.latent_cols, 256))
        for extra_stride in self.extra_strides:
            if self.stride >= extra_stride:
                x = self.conv2d_transpose(x, 256, 4, 2, activation='leaky', bn=bn)
            else:
                break
        x = self.conv2d_transpose(x, 256, 4, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 256, 4, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 128, 4, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 64, 4, 2, activation='leaky', bn=bn)
        x = self.conv2d_transpose(x, 32, 4, 2, activation='leaky', bn=bn)
        ae_d_output = self.conv2d_transpose(x, self.generate_shape[-1], 1, 1, activation='linear')
        return ae_d_input, ae_d_output

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='leaky', kernel_constraint=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_constraint=kernel_constraint,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides, bn=False, activation='leaky'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='leaky'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization(momentum=0.8)(x)

    def kernel_initializer(self):
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def activation(self, x, activation):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        elif activation != 'linear':
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.lvgan_g.summary()
        print()
        self.lvgan_d.summary()
        print()
        self.ae_e.summary()
        print()
        self.ae_d.summary()

