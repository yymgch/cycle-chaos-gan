# utility functions

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mynpdtype = np.float32

# loading MNIST dataset


class TripletDataset:

  def __init__(self, digits, bs, tbs, name_dataset='mnist'):
    ''' loading original dataset, extracting triplet images, and making datasets 
    '''
    self.digits = digits
    self.bs = bs
    self.tbs = tbs
    self.name_dataset = name_dataset
    self.train_images, self.train_labels, self.test_images, self.test_labels = None, None, None, None

    if name_dataset == 'mnist':
      self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_mnist_images()
    elif name_dataset == 'fashion_mnist':
     self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_fashion_mnist_images()

    self.prepare_triplet_images(self.digits)

    self.make_datasets()

  def load_mnist_images(self):
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()  # output is numpy array

    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype(mynpdtype)  # reshape and change data type
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    test_images = test_images.reshape(
        test_images.shape[0], 28, 28, 1).astype(mynpdtype)  # reshape and change data type
    # Normalize the images to [-1, 1]
    test_images = (test_images - 127.5) / 127.5
    return train_images, train_labels, test_images, test_labels

  def load_fashion_mnist_images(self):
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.fashion_mnist.load_data()  # numpy array

    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype(mynpdtype)  # reshape and change data type
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    test_images = test_images.reshape(
        test_images.shape[0], 28, 28, 1).astype(mynpdtype)  #
    # Normalize the images to [-1, 1]
    test_images = (test_images - 127.5) / 127.5
    return train_images, train_labels, test_images, test_labels

  def extract_label_image(self, images, labels, label):
    ind_x = np.where(labels == label)
    images_x = images[ind_x]
    return images_x

  def prepare_triplet_images(self, digits):
    self.train_images_x = self.extract_label_image(
        self.train_images, self.train_labels, digits[0])
    self.train_images_y = self.extract_label_image(
        self.train_images, self.train_labels, digits[1])
    self.train_images_z = self.extract_label_image(
        self.train_images, self.train_labels, digits[2])

    self.test_images_x = self.extract_label_image(
        self.test_images, self.test_labels, digits[0])
    self.test_images_y = self.extract_label_image(
        self.test_images, self.test_labels, digits[1])
    self.test_images_z = self.extract_label_image(
        self.test_images, self.test_labels, digits[2])

  def make_datasets(self):
    self.ds_train_x = tf.data.Dataset.from_tensor_slices(self.train_images_x).shuffle(
        10000).batch(self.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.ds_train_y = tf.data.Dataset.from_tensor_slices(self.train_images_y).shuffle(
        10000).batch(self.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.ds_train_z = tf.data.Dataset.from_tensor_slices(self.train_images_z).shuffle(
        10000).batch(self.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    self.ds_test_x = tf.data.Dataset.from_tensor_slices(self.test_images_x).batch(
        self.tbs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.ds_test_y = tf.data.Dataset.from_tensor_slices(self.test_images_y).batch(
        self.tbs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    self.ds_test_z = tf.data.Dataset.from_tensor_slices(self.test_images_z).batch(
        self.tbs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def generate_images_loop(gen, titles, test_input, n_row=5, loop=1):
    ''' generate images and show them in a loop G(X), G(G(X)), G(G(G(X))), ...
    '''
    # initialization
    generated = test_input
    n_col = 3 * loop + 1
    fig, axes = plt.subplots(n_row, n_col, figsize=(1*n_col, 1.*n_row))

    # generation and display
    for j in range(loop):
        for i in range(3):
            generated = gen(generated, training=False).numpy()
            for r in range(min(n_row, len(test_input))):
                cm = 'gray'
                axes[r][3*j+i+1].imshow(generated[r, :, :, 0]
                                        * 0.5 + 0.5, cmap=cm)
                axes[r][3*j+i+1].axis('off')
                if j == 0 and r == 0:
                    axes[r][3*j+i+1].set_title(titles[i+1])

    # original images
    for r in range(min(n_row, len(test_input))):
        axes[r][0].imshow(test_input[r, :, :, 0] * 0.5 + 0.5, cmap='gray')
        axes[r][0].axis('off')
        if r == 0:
            axes[r][0].set_title(titles[0])

    return fig


def generate_images(gen_1, gen_2, test_input, n_row=5, n_col3=1, labels=['x', 'G(x)', 'F(G(x))']):
  generated_y = gen_1(test_input)
  generated_x = gen_2(generated_y)
  generated_y = generated_y.numpy()
  generated_x = generated_x.numpy()
  n_row = np.min((n_row, 1+test_input.shape[0]//n_col3))
  n_col = 3*n_col3
  fig, axes = plt.subplots(n_row, n_col, figsize=(1*n_col, 1.*n_row))

  for i in range(n_row):
    for j in range(n_col3):
      cm = 'gray'
      axes[i][3*j+0].imshow(test_input[n_col3*i+j, :, :, 0]
                            * 0.5 + 0.5, cmap=cm)
      axes[i][3*j+0].axis('off')
      axes[i][3*j+1].imshow(generated_y[n_col3*i+j]*0.5+0.5, cmap=cm)
      axes[i][3*j+1].axis('off')
      axes[i][3*j+2].imshow(generated_x[n_col3*i+j]*0.5+0.5, cmap=cm)
      axes[i][3*j+2].axis('off')

      if i == 0:
        axes[i][3*j+0].set_title(labels[0])
        axes[i][3*j+1].set_title(labels[1])
        axes[i][3*j+2].set_title(labels[2])

  return fig
