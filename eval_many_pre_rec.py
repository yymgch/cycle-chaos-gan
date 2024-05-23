######
# eval_pre_rec.py
# calculate precision and recall from n-times mapped points
#####
# %%


import umap
from sklearn.decomposition import PCA
from precision_recall import get_pretrained_classifier_model, convert_mnist_to_imagenet_size, knn_precision_recall_features
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from time import time
import model
from util import TripletDataset, generate_images_loop
from parameters import exp_name, name_dataset, DIGITS_PAIR, set_random_seed
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import os
from re import A
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


# %%
# setting matplotlib
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 18

set_random_seed()  # fix random seed of randomとnumpy, tensorflow
data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'pre_rec')

# place of trajectory data
# trj_single_data_dir = os.path.join(data_dir_base, exp_name, 'umap_train')
trj_many_data_dir = os.path.join(data_dir_base, exp_name, 'umap_train_many')


fig_base_dir = './figures'
fig_dir = os.path.join(fig_base_dir, exp_name, 'pre_rec')
os.makedirs(fig_dir, exist_ok=True)

# %%
# GPU detection and setting
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(
            physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

AUTOTUNE = tf.data.AUTOTUNE

tf.config.experimental.enable_tensor_float_32_execution(
    False)  #
tf.keras.backend.set_floatx('float32')
mydtype = tf.float32
mynpdtype = np.float32

# %%

BUFFER_SIZE = 1000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 20
IMG_WIDTH = 28  # not used
IMG_HEIGHT = 28

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

# %%
##### training #####

# loading
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z

# numpy
train_images_x = triplet_dataset.train_images_x
train_images_y = triplet_dataset.train_images_y
train_images_z = triplet_dataset.train_images_z

test_images_x = triplet_dataset.test_images_x
test_images_y = triplet_dataset.test_images_y
test_images_z = triplet_dataset.test_images_z


########################
# precision and recall #
########################


########################################################################
#### recall and precision from single trajectory data ####
########################################################################

# P/R betw single trajectory dataset and test dataset を比較してprecision/recall
# as control，P/R btw train dataset and test dataset is also calculated.
# prepare train data and test data of x, y, and z

n_x = test_images_x.shape[0]
n_y = test_images_y.shape[0]
n_z = test_images_z.shape[0]
n_test = n_x + n_y + n_z

# concatenate test_images
test_images_xyz = np.concatenate(
    (test_images_x, test_images_y, test_images_z), axis=0)
assert test_images_xyz.shape[0] == n_x + n_y + n_z
print(test_images_xyz.shape)
# same number of train images
train_images_xyz = np.concatenate(
    (train_images_x[:n_x], train_images_y[:n_y], train_images_z[:n_z]), axis=0)
print(train_images_xyz.shape)
assert train_images_xyz.shape == test_images_xyz.shape

# %%
# trajectory data Xs.npy
file_path = os.path.join(trj_many_data_dir, 'Xs_used.npy')
Xs_x = np.load(file_path)
file_path = os.path.join(trj_many_data_dir, 'Xs_used_y.npy')
Xs_y = np.load(file_path)
file_path = os.path.join(trj_many_data_dir, 'Xs_used_z.npy')
Xs_z = np.load(file_path)
file_path = os.path.join(trj_many_data_dir, 'steps_used.npy')
steps_used = np.load(file_path)

print(Xs_x.shape)
print(Xs_y.shape)
print(Xs_z.shape)
Xs = np.concatenate([Xs_x, Xs_y, Xs_z], axis=0).reshape(-1,
                                                        len(steps_used), 28, 28, 1)
print(Xs.shape)  # (3197, len(steps_used), 28, 28, 1)
assert Xs.shape[1] == len(steps_used)
assert Xs.shape[0] == n_test

# %% get feature vectors

# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# %% load pre-trained VGG model

vgg_model = get_pretrained_classifier_model()
vgg_model.summary()
# %%
bs = 50


def get_feature_vectors(model, dataset) -> np.ndarray:
  phi = []
  for im_batch in dataset:
    # preprocess
    im_batch = preprocess_input(
        convert_mnist_to_imagenet_size(im_batch, image_size=(224, 224)))

    # get representation (embedding)
    phi_im = vgg_model(im_batch)  # getting feature vectors
    phi.extend(phi_im.numpy())  # type:ignore

  return np.array(phi)  # to numpy array


test_dataset = tf.data.Dataset.from_tensor_slices(test_images_xyz).batch(bs)
phi_test = get_feature_vectors(vgg_model, test_dataset)
print(phi_test.shape)
# train dataset からも同じ数のデータを取り出す
train_dataset = tf.data.Dataset.from_tensor_slices(train_images_xyz).batch(bs)
phi_train = get_feature_vectors(vgg_model, train_dataset)
print(phi_train.shape)

# get feature vecotor of generated data  each step
phi_gen = []
for t in range(Xs.shape[1]):
  ds_gen_t = tf.data.Dataset.from_tensor_slices(Xs[:, t, :]).batch(bs)
  phi_gen_t = get_feature_vectors(vgg_model, ds_gen_t)
  phi_gen.append(phi_gen_t)
  print(f'{t+1}(/{Xs.shape[1]+1}) th time step done.')

# (len(steps_used), n_test, 4096) # 時間ステップが先頭に来ていることに注意
phi_gen = np.array(phi_gen)
print(phi_gen.shape)

assert phi_gen.shape[0] == len(steps_used)
assert phi_gen.shape[1] == n_test
assert phi_gen.shape[2] == 4096
# %% check separation between different classes in the space of feature vectors via PCA

pca = PCA(n_components=2)
phi_concat = np.concatenate([phi_test, phi_train], axis=0)
[phi_test_pca, phi_train_pca] = np.split(
    pca.fit_transform(phi_concat), 2, axis=0)

# plot
#
plt.plot(phi_test_pca[:n_x, 0], phi_test_pca[:n_x, 1],
         'x', label='test_x', markersize=2)
plt.plot(phi_test_pca[n_x:(n_x+n_y), 0],
         phi_test_pca[n_x:(n_x+n_y), 1], 'x', label='test_y', markersize=2)
plt.plot(phi_test_pca[(n_x+n_y):, 0], phi_test_pca[(n_x+n_y)         :, 1], 'x', label='test_z', markersize=2)


plt.plot(phi_train_pca[:n_x, 0], phi_train_pca[:n_x, 1],
         'o', label='train_x', markersize=2)
plt.plot(phi_train_pca[n_x:(n_x+n_y), 0],
         phi_train_pca[n_x:(n_x+n_y), 1], 'o', label='train_y', markersize=2)
plt.plot(phi_train_pca[(n_x+n_y):, 0], phi_train_pca[(n_x+n_y)         :, 1], 'o', label='train_z', markersize=2)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.6))
# save
plt.savefig(os.path.join(fig_dir, 'feature_pca_test_train.png'))

# %% repeating the same using UMAP

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
phi_concat = np.concatenate([phi_test, phi_train], axis=0)
embedding = reducer.fit_transform(phi_concat)
[phi_test_umap, phi_train_umap] = np.split(embedding, 2, axis=0)  # type:ignore

# umap coordinate of generated data
phi_gen_umap = []
for phi_g in phi_gen:
  phi_g_umap = reducer.transform(phi_g)
  phi_gen_umap.append(phi_g_umap)
  print('.', end='')
phi_gen_umap = np.array(phi_gen_umap)  # (len(steps_used), n_test, 2)

# %%

fig, ax = plt.subplots()
ax.plot(phi_test_umap[:n_x, 0], phi_test_umap[:n_x, 1],
        's', label='test_x', markersize=2, color='lightgreen')
ax.plot(phi_test_umap[n_x:(n_x+n_y), 0], phi_test_umap[n_x:(n_x +
        n_y), 1], 's', label='test_y', markersize=2, color='pink')
ax.plot(phi_test_umap[(n_x+n_y):, 0], phi_test_umap[(n_x+n_y)        :, 1], 's', label='test_z', markersize=2, color='cyan')

ax.plot(phi_train_umap[:n_x, 0], phi_train_umap[:n_x, 1], 'o',
        label='train_x', markersize=2, color='olive', alpha=0.4)
ax.plot(phi_train_umap[n_x:(n_x+n_y), 0], phi_train_umap[n_x:(n_x+n_y), 1],
        'o', label='train_y', markersize=2, color='maroon', alpha=0.4)
ax.plot(phi_train_umap[(n_x+n_y):, 0], phi_train_umap[(n_x+n_y):, 1],
        'o', label='train_z', markersize=2, color='blue', alpha=0.4)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
fname = 'feature_umap_test_train'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )
# %%
# plotting test-data and generated data

for t in range(phi_gen_umap.shape[0]):
  step = steps_used[t]

  fig, ax = plt.subplots()

  ax.plot(phi_test_umap[:n_x, 0], phi_test_umap[:n_x, 1],
          's', label='test_x', markersize=2, color='lightgreen')
  ax.plot(phi_test_umap[n_x:(n_x+n_y), 0], phi_test_umap[n_x:(n_x +
          n_y), 1], 's', label='test_y', markersize=2, color='pink')
  ax.plot(phi_test_umap[(n_x+n_y):, 0], phi_test_umap[(n_x+n_y)          :, 1], 's', label='test_z', markersize=2, color='cyan')

  ax.plot(phi_gen_umap[t, :, 0], phi_gen_umap[t, :, 1], 'o', label='gen',  # type:ignore
          markersize=2, color='purple', alpha=0.3)  # type:ignore

  # ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
  ax.text(0.05, 0.85, f'n={step}', transform=ax.transAxes, fontsize=22)
  os.makedirs(os.path.join(fig_dir, f'umap_test_gen_many'), exist_ok=True)
  fname = f'feature_umap_test_gen_many_step{step}'
  fig.savefig(os.path.join(
      fig_dir, 'umap_test_gen_many/'+fname+'.png'), dpi=360)
  fig.savefig(os.path.join(
      fig_dir, 'umap_test_gen_many/'+fname+'.tiff'), dpi=360)
  fig.savefig(os.path.join(fig_dir, 'umap_test_gen_many/'+fname+'.pdf'), )
# %%


# from the result of UMAP, it is confirmed that X, Y, and Z are well separated in the feature space, which means that
# the difference of classes of X, Y, and Z is not lost in the feature space by VGG.


# %% precision/recall
# Now, we have phi_train, phi_test, phi_gen. Next we will calculate precision and recall.
k_nn = 7  # number of nearest neighbors. based on test-train data, we use 7.

# precision and recall between phi_test and phi_train

state_test_train = knn_precision_recall_features(phi_test, phi_train, nhood_sizes=[k_nn],
                                                 row_batch_size=1000, num_gpus=1)
print(state_test_train)

pre_test_gen_t = []
rec_test_gen_t = []
for t in range(phi_gen.shape[0]):
  state_test_gen_i = knn_precision_recall_features(phi_test, phi_gen[t], nhood_sizes=[k_nn],
                                                   row_batch_size=1000, num_gpus=1)
  pre_test_gen_t.append(state_test_gen_i['precision'][0])
  rec_test_gen_t.append(state_test_gen_i['recall'][0])
  print(f'{t+1}th step done.')


# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(steps_used, pre_test_gen_t, 'ro-',
        label='precision(test, gen)', markersize=6)
ax.plot(steps_used, rec_test_gen_t, 'b^-',
        label='recall(test, gen)', markersize=6)
ax.set_xlabel('n')
# use logscale for x-axis
ax.set_xscale('symlog')
ax.grid()
# set position of legend outside of the plot
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
ax.legend()
fig.tight_layout()
fname = 'precision_recall_vs_steps'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )
