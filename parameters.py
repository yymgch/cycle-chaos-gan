# store global parameters for common use

import random
import numpy as np
import tensorflow as tf
exp_name = str(1000)+'EP_G_10_D_10_10_10'
# exp_name = 'Fashion_MNIST'+ str(2000)+ 'EP_G_10_D_10_10_10'
# name_dataset = 'fashion_mnist'
name_dataset = 'mnist'
DIGITS_PAIR = [0, 1, 2]
# DIGITS_PAIR = [0, 7, 8] # for fasion_mnist
EPOCHS = 1000


def set_random_seed(seed=12345):
    # Tensorflow
    tf.random.set_seed(seed+1)
    # Numpy
    np.random.seed(seed+2)
    # Python random module
    random.seed(seed+3)
