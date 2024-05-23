## model implementation ##


from time import time
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


# Generator

def make_generator_model(n_image_channel=1):
    '''
        Simple encoder_decoder model
        The first half uses Conv2D and Strides to reduce the image size
        The second half enlarges the image size
    '''
    # Sequential
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.LayerNormalization(axis=[1, 2])) # type:ignore
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.LayerNormalization(axis=[1, 2])) # type:ignore
    model.add(layers.LeakyReLU())

    # insert one ResNet Block
    model.add(ResnetBlockV2(filter_size=128))

    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2DTranspose(
        128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.LayerNormalization(axis=[1, 2])) # type:ignore
    model.add(layers.LeakyReLU())

    #
    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.LayerNormalization(axis=[1, 2]))# type:ignore
    model.add(layers.LeakyReLU())

    #
    model.add(layers.Conv2DTranspose(n_image_channel, (5, 5), strides=(
        2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 1)

    return model


# ResNet Block (v2)

class ResnetBlockV2(tf.keras.layers.Layer):

    def __init__(self, kernel_size=(3, 3), filter_size=16, stride=1, dif_fsize=False):
        ''' args:
            kernel_size: kernel_size. default is (3,3)
            filter_size: numbers of output filters.
            stride: scalar. If this is not 1, skip connection are replaced to 1x1 convolution.
            dif_fsize: True if the numbers of input filter and output filter are different
        '''
        super(ResnetBlockV2, self).__init__(name='')

        if stride == 1:
            strides = (1, 1)
        else:
            strides = (stride, stride)

        self.bn2a = tf.keras.layers.LayerNormalization()
        self.conv2a = tf.keras.layers.Conv2D(
            filter_size, kernel_size, strides=strides, padding='same')

        self.bn2b = tf.keras.layers.LayerNormalization()
        self.conv2b = tf.keras.layers.Conv2D(
            filter_size, kernel_size, strides=(1, 1), padding='same')

        self.use_identity_shortcut = (stride == 1) and not dif_fsize
        if not self.use_identity_shortcut:
            self.conv2_sc = tf.keras.layers.Conv2D(
                filter_size, (1, 1), strides=strides, padding='same')

    def call(self, input_tensor, training=False):
        x = self.bn2a(input_tensor, training=training)
        x1 = tf.nn.relu(x)  #
        x = self.conv2a(x1)  #

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        if self.use_identity_shortcut:
            skip = input_tensor
        else:
            skip = self.conv2_sc(x1)
        x += skip
        return x


# Discriminator model

def make_discriminator_model_addDropout():
    '''
    '''
    model = tf.keras.Sequential()

    # assert model.output_shape == (None, 28, 28, 1)
    model.add(layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.LayerNormalization(axis=[1, 2])) #type:ignore
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.LayerNormalization(axis=[1, 2])) #type:ignore
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(layers.LayerNormalization(axis=[1, 2])) #type:ignore
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2),
              padding='same', use_bias=False, activation='tanh'))
    model.add(layers.GlobalAveragePooling2D())
    # assert model.output_shape == (None, 28, 28, 1)
    model.add(layers.Dense(1, use_bias=True))

    return model


# loss functions
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# for Discriminator


def discriminator_los(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


# discriminator loss for cycle gan
def discriminator_loss(real, generated_g, generated_f):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss_g = loss_obj(tf.zeros_like(generated_g), generated_g)
    generated_loss_f = loss_obj(tf.zeros_like(generated_f), generated_f)

    total_disc_loss = real_loss + (generated_loss_g + generated_loss_f) * 0.5

    return total_disc_loss


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


# cycle-consistency loss
def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

# identity loss (unused)


def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


# save weights

def models_save_weights(dir_ckpt, name_ckpt, gen_g, gen_f, disc_x, disc_y, disc_z):
    # def models_save_weights(dir_ckpt, name_ckpt, gen_g, gen_f, disc_x, disc_y):
    gen_g.save_weights(os.path.join(dir_ckpt, name_ckpt+"-gen_g"))
    gen_f.save_weights(os.path.join(dir_ckpt, name_ckpt+"-gen_f"))
    disc_x.save_weights(os.path.join(dir_ckpt, name_ckpt+"-disc_x"))
    disc_y.save_weights(os.path.join(dir_ckpt, name_ckpt+"-disc_y"))
    disc_z.save_weights(os.path.join(dir_ckpt, name_ckpt+"-disc_z"))
