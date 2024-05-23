# calculation of Lyapunov spectrum via getting Jacobian matrix and QR decomposition
# %%
from parameters import exp_name, name_dataset, DIGITS_PAIR, set_random_seed
from util import TripletDataset, generate_images
from typing import Any, List, Tuple
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


# import tensorflow_datasets as tfds

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
    False)  # tf32
tf.keras.backend.set_floatx('float32')
mydtype = tf.float32
mynpdtype = np.float32


BUFFER_SIZE = 1000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 200
IMG_WIDTH = 28  # not used
IMG_HEIGHT = 28
# DIGITS_PAIR = [0, 1, 2]
LAMBDA = 10  # weight for consistency loss

# exp_name = str(1000)+'EP_G_10_D_10_10_10'
data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'lyap')

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

set_random_seed()  # fix seeds of random, numpy, tensorflow

# %%

# loading dataset
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z

# %%

generator_g = model.make_generator_model()
generator_f = model.make_generator_model()


discriminator_x = model.make_discriminator_model_addDropout()
discriminator_y = model.make_discriminator_model_addDropout()
discriminator_z = model.make_discriminator_model_addDropout()

# %%
# check
# 0->1
im = next(iter(ds_train_x))
g_x = generator_g(im)
print(f'generator output shape:{g_x.shape}')
d = discriminator_y(g_x)
print(f'discriminator output shape:{d.shape}')
fg_x = generator_f(g_x)
d_y = discriminator_x(fg_x)

fig = generate_images(generator_g, generator_f, im,
                      labels=['x', 'G(x)', 'F(G(x))'])
# plt.show()
# %%
# 1->0
im = next(iter(ds_train_y))
f_y = generator_f(im)
print(f'generator output shape:{f_y.shape}')
d = discriminator_x(f_y)
print(f'discriminator output shape:{d.shape}')
gf_y = generator_g(f_y)
d_x = discriminator_y(gf_y)

fig = generate_images(generator_f, generator_g, im,
                      labels=['x', 'F(x)', 'G(F(x))'])
# plt.show()


# %%
# Loss function

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_z_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# checkpoints
checkpoint_path = os.path.join("./checkpoints", exp_name)
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           discriminator_z=discriminator_z,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer,
                           discriminator_z_optimizer=discriminator_z_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# model.models_save_weights(checkpoint_path, 'weights-0', generator_g, generator_f, discriminator_x, discriminator_y, discriminator_z)


# %%
# @tf.function
# input and output are 784-dim vectors
def g_784(x):
    x = tf.reshape(x, (-1, 28, 28, 1))
    y = generator_g(x)
    y = tf.reshape(y, (-1, 784))
    return y


@tf.function
def g3(x):
    '''three transformations of generator_g'''
    y = generator_g(x)
    z = generator_g(y)
    return generator_g(z), y, z


@tf.function
def g3_784(x):
    x = tf.reshape(x, (-1, 28, 28, 1))
    y = generator_g(x)

    # y = tf.reshape(y, (-1, 28, 28, 1))
    z = generator_g(y)

    # z = tf.reshape(z, (-1, 28, 28, 1))
    x_new = generator_g(z)

    x_new = tf.reshape(x_new, (-1, 784))
    y = tf.reshape(y, (-1, 784))
    z = tf.reshape(z, (-1, 784))
    return x_new, y, z


def iterate_g3(x0, t_max=10, g3=g3, transient=0, stack=True):
    ''' iterate t_max times with g3'''
    x = x0  #
    X = []  #
    Y = []
    Z = []
    for t in range(transient):
        # print(f't={t}')
        x_new, y_new, z_new = g3(x)  # 1回反復
        x = x_new  # x 更新
    # X.append(x)
    for t in range(transient, t_max):
        # print(f't={t}')
        x_new, y_new, z_new = g3(x)
        X.append(x_new)  # 記録する．
        Y.append(y_new)
        Z.append(z_new)
        x = x_new

    if stack:
        X = tf.stack(X, axis=1)  # リストから(bs, len_seq, h,w,c) の形のテンソルにして返す．
        Y = tf.stack(Y, axis=1)
        Z = tf.stack(Z, axis=1)
    return X, Y, Z


def iterate_g(x0, t_max=10, g=generator_g, transient=0, stack=True) -> tf.Tensor:
    ''' iterate g t_max times'''
    x = x0  # initial value
    X = []  #
    g_func = tf.function(g)
    for t in range(transient):
        # print(f't={t}')
        x_new = g_func(x)  #
        x = x_new  #
    # X.append(x)
    for t in range(transient, t_max):
        # print(f't={t}')
        x_new = g_func(x)
        X.append(x_new)  #
        x = x_new

    # if stack:
    X = tf.stack(X, axis=1)  # (bs, len_seq, h,w,c)
    return X


# %% check
x0 = next(iter(ds_test_x))  # (bs, 28, 28, 1)

#  X.shape = (bs, t_max-transient, 28,28,1)
X, Y, Z = iterate_g3(x0, t_max=200, transient=100)

print(X.shape)  # (bs, t_max-transient, 28,18,1)　# type:ignore

# %%
t_max = 4000
transient = 2000
t_len = t_max-transient
n_batch = ds_test_x.cardinality().numpy()  # get size


# %%
def trajectories_from_ds(iteration, ds, t_max, transient=0):
    ''' Generate trajectories for each data in ds using the given iteration function.
    The output is a tensor of each image (h,w,c) flattened to 1 dimension.
    Done for each mini-batch.
        iteration: the iteration function to use
        ds: the dataset, passing (bs, h,w,c) data
        t_max: the number of iterations
        transient: the initial transient
        returns:
        Xs, Ys : arrays of images with size (n_data_dataset, seq_len, h*w*c)
    '''
    n_batch = ds.cardinality().numpy()
    Xs = []
    # Ys = []
    # Zs = []

    for n, init_cond in enumerate(ds):
        X = iteration(init_cond, t_max=t_max, transient=transient)
        Xs.append(X.numpy())
        print(f'minibatch:{n+1}/{n_batch} done')
    # (n_test, t_length, 28*28)
    nh = X.shape[2]
    nw = X.shape[3]
    nc = X.shape[4]
    # (bs, t_len, 28,28,1) to (bs, t_len, 784,)
    Xs = np.concatenate(Xs, axis=0).reshape((-1, t_len, nh*nw*nc))
    return Xs


# %%
# Xs, Ys, Zs = trajectories_from_ds(iterate_g, ds_test_x, t_max=t_max, transient=transient)
Xs = trajectories_from_ds(iterate_g, ds_test_x,
                          t_max=t_max, transient=transient)

print(Xs.shape)  # (980, t_len, 784)

# %%
# 保存先
os.makedirs(data_dir_base, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# %%

np.save(os.path.join(data_dir, 'Xs.npy'), Xs)

# %%

# Xs = np.load( os.path.join(data_dir, 'Xs.npy'))


# %% Lyapnov spectrum
# Xs contains trajectories．Calculate Lyapunov spectrum

# get Jacobians for each X
bs_about = 100  # 一度にいくつ処理するか
n_split = Xs.shape[0] // bs_about  #

# X_split is a list, whoose elements have (bs,len_seq, 784) shape
X_split = np.array_split(Xs, n_split, axis=0)

# %%


@tf.function
def get_jacobian(x):
    '''
    Calculate the Jacobian matrix for a set of images by applying one iteration of the mapping.
    x: Matrix containing a batch of images (bs, 784)
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        x_next = g_784(x)   # one iteration
    J = tape.batch_jacobian(x_next, x, parallel_iterations=100,
                            experimental_use_pfor=True)  # get (nb_about, 784,784)
    return J

# QR decomposition and get diagonal value and its logarithm


@tf.function
def qr_onestep(J, Q):
    A = tf.matmul(J, Q)
    [Q, R] = tf.linalg.qr(A)
    log_abs_diag_r = tf.math.log(tf.abs(tf.linalg.diag_part(R)))  # (bs, 784)
    return Q, log_abs_diag_r


# %%
li_logrs = []  # list of logarithm of r

for ii, xb in enumerate(X_split):
    print(f'{ii}-th batch')
    Q = tf.eye(784, batch_shape=[xb.shape[0]],
               dtype=mydtype)  # For QR decomposition
    # xb shape is (bs,len_seq, 784)
    # to tf.tensor and split
    X_t = tf.unstack(xb, axis=1)  # each element of list have (bs, 784)

    print(f'x_t len: {len(X_t)}')
    print(f'X_t[0].shape {X_t[0].shape}')  # (bs,784)
    logrs = []
    startt = time.time()
    for tt, x in enumerate(X_t):
        J = get_jacobian(x)  # (bs, 784, 784)
        Q, log_abs_diag_r = qr_onestep(J, Q)
        print(f'\r time {tt} jacobian calculated ', end='')
        print(f'and qr decomposition was performed ', end='')
        print(f'{time.time()-startt} was elapsed since the start of this batch')
        # print("")
        # print(f'Q.shape {Q.shape}')
        logrs.append(log_abs_diag_r)
    logrs = tf.stack(logrs, axis=1)  # (bs, len, 784)
    li_logrs.append(logrs)


logrs = tf.concat(li_logrs, axis=0)
lsps = tf.reduce_mean(logrs, axis=1)  # (bs,784)


# %%
np.save(os.path.join(data_dir, 'logrs.npy'), logrs.numpy())  # type:ignore
np.save(os.path.join(data_dir, 'lsps.npy'), lsps.numpy())

# %% save text file for logrs
with open(os.path.join(data_dir, 'lsps.txt'), 'w') as f:
    for l in lsps.numpy():
        f.write(' '.join(map(str, l))+'\n')
