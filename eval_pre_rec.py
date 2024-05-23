######
# eval_pre_rec.py
# calculate precision and recall from single trajectory of generated data
#####
# %%
#########


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

set_random_seed()  # randomとnumpy, tensorflowのseedを固定
data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'pre_rec')

# trajectory が保存されたfileの置き場
trj_single_data_dir = os.path.join(data_dir_base, exp_name, 'umap_train')
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
##### training data #####


# loading dataset
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z

# numpy array
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

# Comparing single trajectory data with test data, and evaluate precision and recall
# As a control, we also compare train data with test data.
# Prepare the same number of training data or generated data as the total number of test data of x, y, and z.

n_x = test_images_x.shape[0]
n_y = test_images_y.shape[0]
n_z = test_images_z.shape[0]
n_test = n_x + n_y + n_z

# combine test_images
test_images_xyz = np.concatenate(
    (test_images_x, test_images_y, test_images_z), axis=0)
assert test_images_xyz.shape[0] == n_x + n_y + n_z
print(test_images_xyz.shape)
# prepare same number of training data as the total number of test data of x, y, and z.
train_images_xyz = np.concatenate(
    (train_images_x[:n_x], train_images_y[:n_y], train_images_z[:n_z]), axis=0)
print(train_images_xyz.shape)
assert train_images_xyz.shape == test_images_xyz.shape

# loading trajectory data
file_path = os.path.join(trj_single_data_dir, 'Xs.npy')
Xs = np.load(file_path)
print(Xs.shape)
# trajectory (remove images of initial value t=0 because it is test image )
X = Xs[:, 1:(n_test+1)]

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
# train dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images_xyz).batch(bs)
phi_train = get_feature_vectors(vgg_model, train_dataset)
print(phi_train.shape)

# get feature vectors of generated data

ds_gen = tf.data.Dataset.from_tensor_slices(X[0, :]).batch(bs)
phi_gen = get_feature_vectors(vgg_model, ds_gen)
print(phi_gen.shape)


# %% PCA

pca = PCA(n_components=2)
phi_concat = np.concatenate([phi_test, phi_train], axis=0)
[phi_test_pca, phi_train_pca] = np.split(
    pca.fit_transform(phi_concat), 2, axis=0)

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


# upam coordinate of generated data in feature space
phi_gen_umap = reducer.transform(phi_gen)

# %%
fig, ax = plt.subplots()
ax.plot(phi_test_umap[:n_x, 0], phi_test_umap[:n_x, 1],
        's', label='test_x', markersize=2, color='lightgreen')
ax.plot(phi_test_umap[n_x:(n_x+n_y), 0], phi_test_umap[n_x:(n_x +
        n_y), 1], 's', label='test_y', markersize=2, color='pink')
ax.plot(phi_test_umap[(n_x+n_y):, 0], phi_test_umap[(n_x+n_y)
        :, 1], 's', label='test_z', markersize=2, color='cyan')

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

# test-data and generated data
fig, ax = plt.subplots()
ax.plot(phi_test_umap[:n_x, 0], phi_test_umap[:n_x, 1],
        's', label='test_x', markersize=2, color='lightgreen')
ax.plot(phi_test_umap[n_x:(n_x+n_y), 0], phi_test_umap[n_x:(n_x +
        n_y), 1], 's', label='test_y', markersize=2, color='pink')
ax.plot(phi_test_umap[(n_x+n_y):, 0], phi_test_umap[(n_x+n_y)
        :, 1], 's', label='test_z', markersize=2, color='cyan')

ax.plot(phi_gen_umap[:, 0], phi_gen_umap[:, 1], 'o', label='gen',  # type:ignore
        markersize=2, color='purple', alpha=0.3)  # type:ignore

ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
fname = 'feature_umap_test_gen'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )


# from the result of UMAP, it is clear that X, Y, and Z are well separated
# in the feature space, which means that the difference between the classes
# of XYZ is not lost by VGG.

# %% precision/recall
# Now, we have phi_train, phi_test, phi_gen. Next we will calculate precision and recall.
k_nn = 10

# precision and recall between phi_test and phi_train

state_test_train = knn_precision_recall_features(phi_test, phi_train, nhood_sizes=[k_nn],
                                                 row_batch_size=1000, num_gpus=1)
print(state_test_train)

state_test_gen = knn_precision_recall_features(phi_test, phi_gen, nhood_sizes=[k_nn],
                                               row_batch_size=1000, num_gpus=1)
print(state_test_gen)

# compare result of different k_nn

k_nns = np.arange(1, 11)
pre_test_train_ks = []
rec_test_train_ks = []

pre_test_gen_ks = []
rec_test_gen_ks = []

for k_nn in k_nns:
  state = knn_precision_recall_features(phi_test, phi_train, nhood_sizes=[k_nn],
                                        row_batch_size=1000, num_gpus=1)
  pre_test_train_ks.append(state['precision'][0])
  rec_test_train_ks.append(state['recall'][0])

  state_gen = knn_precision_recall_features(phi_test, phi_gen, nhood_sizes=[k_nn],
                                            row_batch_size=1000, num_gpus=1)
  pre_test_gen_ks.append(state_gen['precision'][0])
  rec_test_gen_ks.append(state_gen['recall'][0])

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_nns, pre_test_train_ks, 'ro--',
        label='precision(test, train)', markersize=6)
ax.plot(k_nns, rec_test_train_ks, 'b^--',
        label='recall(test, train)', markersize=6)
ax.plot(k_nns, pre_test_gen_ks, 'ro-',
        label='precision(test, gen)', markersize=6)
ax.plot(k_nns, rec_test_gen_ks, 'b^-', label='recall(test, gen)', markersize=6)
ax.set_xlabel('k')
ax.grid()
# set position of legend outside of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
fig.tight_layout()
fname = 'precision_recall_different_k'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )


# %% p/r for 100 trajectories

# get phi_gen for 100 trajectories
phi_gens = []
for i in range(X.shape[0]):
  ds_gen = tf.data.Dataset.from_tensor_slices(X[i, :]).batch(bs)
  phi_gen = get_feature_vectors(vgg_model, ds_gen)
  phi_gens.append(phi_gen)
  # print(phi_gen)
  print(f'{i+1}th trajectory done.')
  # print(phi_gens[0])
  # print(phi_gens[i])

phi_gens = np.array(phi_gens)
print(phi_gens.shape)  # (100, n_test, 4096)

# %%

k_nns = np.arange(1, 11)

pre_test_gens = []
rec_test_gens = []
for i in range(X.shape[0]):
  pre_i = []
  rec_i = []
  for k_nn in k_nns:

    state_gens = knn_precision_recall_features(phi_test, phi_gens[i], nhood_sizes=[k_nn],
                                               row_batch_size=1000, num_gpus=1)
    pre_i.append(state_gens['precision'][0])
    rec_i.append(state_gens['recall'][0])

    print(
        f'{i+1}th trajectory, k={k_nn} done. precision={pre_i[-1]}, recall={rec_i[-1]}')
    # print(pre_i)
  pre_test_gens.append(np.array(pre_i))
  rec_test_gens.append(np.array(rec_i))

pre_test_gens = np.array(pre_test_gens)
rec_test_gens = np.array(rec_test_gens)
print(pre_test_gens.shape)
print(rec_test_gens.shape)
# %% save
os.makedirs(data_dir, exist_ok=True)
np.save(os.path.join(data_dir, 'pre_test_gens.npy'), pre_test_gens)
np.save(os.path.join(data_dir, 'rec_test_gens.npy'), rec_test_gens)
# %%
# mean and std of precision and recall


m_pre_test_gens = np.mean(pre_test_gens, axis=0)
m_rec_test_gens = np.mean(rec_test_gens, axis=0)
std_pre_test_gens = np.std(pre_test_gens, axis=0)
std_rec_test_gens = np.std(rec_test_gens, axis=0)

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_nns, pre_test_train_ks, 'ro--',
        label='precision(test, train)', markersize=6)
ax.plot(k_nns, rec_test_train_ks, 'b^--',
        label='recall(test, train)', markersize=6)

ax.errorbar(k_nns, m_pre_test_gens, yerr=std_pre_test_gens,
            fmt='ro-', label='precision(test, gen)', markersize=6)
ax.errorbar(k_nns, m_rec_test_gens, yerr=std_rec_test_gens,
            fmt='b^-', label='recall(test, gen)', markersize=6)

ax.set_xlabel('k')
ax.grid()
# set position of legend outside of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.6))
fig.tight_layout()
fname = 'precision_recall_mean_with_k'

fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )


# %% k=7のときのprecision/recallのヒストグラムをみる

k_nn = 7
ind = np.where(k_nns == k_nn)[0][0]
print(ind)
pre_test_gens_k7 = pre_test_gens[:, ind]
rec_test_gens_k7 = rec_test_gens[:, ind]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pre_test_gens_k7, bins=20, alpha=0.5, color='r', label='precision')
ax.hist(rec_test_gens_k7, bins=20, alpha=0.5, color='b', label='recall')
ax.legend()
ax.set_xlabel('value')
ax.set_ylabel('frequency')
ax.grid()
fig.tight_layout()
fname = 'precision_recall_k7_hist'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), )

# %%
