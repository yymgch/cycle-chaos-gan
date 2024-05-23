# draw model graph
# pydot is needed

#%%
# %%
from parameters import exp_name, name_dataset, DIGITS_PAIR, set_random_seed, EPOCHS
from util import TripletDataset
import requests
import json
import model
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import os
from re import A
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'training')
# %%

###### GPU #####

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
  for k in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[k], True)
    print('memory growth:', tf.config.experimental.get_memory_growth(
        physical_devices[k]))
else:
  print("Not enough GPU hardware devices available")

AUTOTUNE = tf.data.AUTOTUNE

#%% test

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape=(784,))
hidden_layer = Dense(128, activation='relu')(input_layer)
output_layer = Dense(10, activation='softmax')(hidden_layer)

modela = Model(inputs=input_layer, outputs=output_layer)

#%%
from tensorflow.keras.utils import plot_model

plot_model(modela,  show_shapes=True, show_layer_names=True)



# %% cnn

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

input_layer = Input(shape=(28, 28, 1))
conv_layer = Conv2D(32, (3, 3), activation='relu')(input_layer)
pooling_layer = MaxPooling2D((2, 2))(conv_layer)
flatten_layer = Flatten()(pooling_layer)
output_layer = Dense(10, activation='softmax')(flatten_layer)

modelb = Model(inputs=input_layer, outputs=output_layer)

from tensorflow.keras.utils import plot_model

plot_model(modelb,  show_shapes=True, show_layer_names=True, expand_nested=True)



# %%

# BUFFER_SIZE = 1000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 20
IMG_WIDTH = 28  # not used
IMG_HEIGHT = 28
LAMBDA = 10  # weight for consistency loss

# exp_name = str(1000)+'EP_G_10_D_10_10_10'

print(f'exp_name:{exp_name}')

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

set_random_seed()
# %%
##### MNIST and Fashion MNIST dataset #####
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z

# %%
##### model #####


generator_g = model.make_generator_model()
generator_f = model.make_generator_model()


discriminator_x = model.make_discriminator_model_addDropout()
discriminator_y = model.make_discriminator_model_addDropout()
discriminator_z = model.make_discriminator_model_addDropout()

# %%
##########
# set input shape and initialize weights
sample_x = next(iter(ds_train_x))  # get sample data
_ = generator_g(sample_x)  # warm up
print(_.shape)
_ = generator_f(sample_x)
print(_.shape)
_ = discriminator_x(sample_x)
print(_.shape)
_ = discriminator_y(sample_x)
print(_.shape)
_ = discriminator_z(sample_x)
print(_.shape)
##########
# %%
fig_dir = './figures/'+ exp_name + '/graph'
os.makedirs(fig_dir, exist_ok=True)

fname = os.path.join(fig_dir,'model_graph_generator')
plot_model(generator_g, to_file=fname+'.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=360)
plot_model(generator_g, to_file=fname+'.pdf', show_shapes=True, show_layer_names=True, expand_nested=True)
#plot_model(generator_g, to_file=fname+'.tiff', show_shapes=True, show_layer_names=True, expand_nested=True)

# %%
fname = os.path.join(fig_dir,'model_graph_discriminator')
plot_model(generator_g, to_file=fname+'.pdf', show_shapes=True, show_layer_names=True, expand_nested=True)
plot_model(generator_g, to_file=fname+'.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=360)


# %%
