
# %% this code calculate the largest Lyapunov exponent from the direct evolution of the orbit
# %%
from sklearn.linear_model import LinearRegression
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
    False)  #

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

set_random_seed()  # fix seeds of randomとnumpy, tensorflow

# %%


# load dataset
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


###############
###############
# parameters
transient = 2000
n_step = 30  # length of time step
eps_perturb = 1e-4  # strength of perturbation

# %%
# load logrs.npy
lsps = np.load(os.path.join(data_dir, 'lsps.npy'))
m_lsps = np.mean(lsps, axis=0)
LE_est = m_lsps[0]
print(f'the estimated largest Lyapunov exponent:{LE_est}')

# get trajectories
X_start = []
for n, x0 in enumerate(ds_test_x):
  x = tf.reshape(x0, (-1, 28, 28, 1))
  for i in range(2000):
    x_new = generator_g(x)
    x = x_new
  X_start.append(x)
  print(f'x_start:{n} done')


X_start = np.concatenate(X_start, axis=0).reshape((-1, 784))  # (980, 784)
print(f'X_start.shape:{X_start.shape}')

# %%
#############################
# random perturbation version
#############################

# setting parameters
# How many original trajectories are included in one batch.
n_orig_in_batch = 5
n_perturb = 10  # How many perturbations are applied to one trajectory.

bs = n_orig_in_batch*n_perturb


# %%


ds_initial = tf.data.Dataset.from_tensor_slices(X_start).batch(n_orig_in_batch)
X_perturbed = []
X_no_perturb = []
with tf.device('/gpu:0'):
  for n, x0 in enumerate(ds_initial):
    # make perturbed initial conditions
    # copy x0
    n_orig = x0.shape[0]
    x0c = tf.tile(x0, [n_perturb, 1])
    assert x0c.shape == (n_orig*n_perturb, 784)  # type:ignore
    x0_perturb = x0c + eps_perturb * \
        tf.random.normal(x0c.shape, dtype=np.float32)  # type:ignore
    # [-1,1] clipping
    x0_perturb = tf.clip_by_value(x0_perturb, -1, 1)
    # print(x0_perturb.shape)
    # iteration
    x = tf.reshape(x0_perturb, (-1, 28, 28, 1))
    x_perturbed = [x]
    for n in range(n_step):
      x_new = generator_g(x)
      x_perturbed.append(x_new)
      x = x_new
    x_perturbed = np.array(tf.stack(x_perturbed, axis=1)).reshape(
        (n_perturb, n_orig, -1, 28*28))
    # (n_orig, n_perturb, n_step+1, 28*28)
    x_perturbed = x_perturbed.transpose((1, 0, 2, 3))
    # print(x_perturbed.shape)
    X_perturbed.append(x_perturbed)
    print(f'minibatch:{n+1} done')

    # no perturbation
    x = tf.reshape(x0, (-1, 28, 28, 1))
    x_no_perturb = [x]
    for n in range(n_step):
      x_new = generator_g(x)
      x_no_perturb.append(x_new)
      x = x_new
    x_no_perturb = np.transpose(
        np.array(x_no_perturb).reshape(n_step+1, -1, 28*28), (1, 0, 2,))
    X_no_perturb.append(x_no_perturb)

  # (980, n_perturb, n_step+1, 28*28)
  X_perturbed = np.concatenate(X_perturbed, axis=0)
  X_no_perturb = np.concatenate(X_no_perturb, axis=0)  # (980, n_step+1, 28*28)
# %%
print(f'X_perturbed.shape:{X_perturbed.shape}')
print(f'X_no_perturbed.shape:{X_no_perturb.shape}')

# %%
X_no_perturb_bc = np.expand_dims(X_no_perturb, 1)
print(f'X_no_perturb.shape:{X_no_perturb.shape}')
print(f'X_no_perturb_bc.shape:{X_no_perturb_bc.shape}')

# %% difference between perturbed and non-perturbed
diff = X_perturbed - X_no_perturb_bc
abs_diff = np.sqrt(np.sum(diff*diff, axis=3))  # dimension direction

# %%

# matplotlib setting
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

fig_dir = './figures/' + exp_name + '/lyap'
os.makedirs(fig_dir, exist_ok=True)

# %%

abs_diff_t = abs_diff.transpose(1, 2, 0)
mean_log_abs_diff = np.mean(np.log(abs_diff_t), axis=(2))
fig, ax = plt.subplots()
# ax.grid()
# set xticks
ax.set_xticks(np.arange(0, n_step+1, 5))

ax.plot(abs_diff_t[0], alpha=0.2, color='gray', linewidth=0.5)
ax.plot(np.exp(mean_log_abs_diff[0]), 'o-', label='mean of log', color='blue')
# ax.plot( np.mean(abs_diff_t[0], axis=1), label='mean', color='red')
ax.set_yscale('log')

# LE_est = 0.341
x_val = np.arange(0, 10)
y_val = np.exp(LE_est*x_val - 2)
ax.plot(x_val, y_val, color='green')

fname = 'direct_lyap_random_perturb'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'))
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360)


###########
# perturbation along the direction of the attractor
###########
# %%
# get nearest neighbor of each point and the distance to it
print(X_start.shape)
# matrix of squared distance
sqX_start = np.sum(X_start**2, axis=1)
D = np.expand_dims(sqX_start, 0) + np.expand_dims(sqX_start, 1)
D -= 2 * np.matmul(X_start, X_start.T)
# %%
np.fill_diagonal(D, np.inf)
minD = np.min(D, axis=1)
arg_minD = np.argmin(D, axis=1)
# print( np.sqrt(minD))

X_start_nb = X_start[arg_minD]  # nearest neighbor
d = (X_start_nb - X_start)
d_norm = np.sqrt(np.sum(d*d, axis=1))

X_start_perturb = X_start + eps_perturb * d/np.expand_dims(d_norm, 1)

np.sqrt(np.sum((X_start - X_start_perturb)**2, axis=1))

# %% from X_start and X_start_perturb
X_no_perturb = []
X_perturb = []
ds_x_and_xp = tf.data.Dataset.from_tensor_slices(
    (X_start, X_start_perturb)).batch(50)
for n, (x_s, x_s_p) in enumerate(ds_x_and_xp):
  x = tf.reshape(x_s, (-1, 28, 28, 1))
  x_p = tf.reshape(x_s_p, (-1, 28, 28, 1))
  x_no_perturb = [x]
  x_perturb = [x_p]
  for i in range(n_step):
    x_new = generator_g(x)
    x_p_new = generator_g(x_p)
    x_no_perturb.append(x_new)
    x_perturb.append(x_p_new)
    x = x_new
    x_p = x_p_new
  x_no_perturb = np.transpose(
      np.array(x_no_perturb), (1, 0, 2, 3, 4)).reshape(-1, n_step+1, 28*28)
  x_perturb = np.transpose(
      np.array(x_perturb), (1, 0, 2, 3, 4)).reshape(-1, n_step+1, 28*28)
  X_no_perturb.append(x_no_perturb)
  X_perturb.append(x_perturb)
  print(f'batch:{n} done')

X_no_perturb = np.concatenate(X_no_perturb, axis=0)
X_perturb = np.concatenate(X_perturb, axis=0)
# %%
D = X_no_perturb - X_perturb
abs_D = np.sqrt(np.sum(D*D, axis=2))
mean_log_abs_diff = np.mean(np.log(abs_D), axis=0)

fig, ax = plt.subplots()
ax.set_xticks(np.arange(0, n_step+1, 5))
ax.plot(abs_D.T, alpha=0.2, color='gray', linewidth=0.5)
ax.plot(np.exp(mean_log_abs_diff), 'o-', label='mean', color='blue')
ax.set_yscale('log')
ax.set_xlabel('n')
ax.set_ylabel('distance')

# LE_est = 0.341
x_val = np.arange(0, 21)
y_val = np.exp(LE_est*x_val - 6)
ax.plot(x_val, y_val, color='green')
fig.tight_layout
fname = 'direct_lyap_tangent_perturb'
fig.savefig(os.path.join(fig_dir, fname+'.png'), dpi=360, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), dpi=360, bbox_inches='tight')
# plt.plot(np.exp(mean_log_abs_diff))

# %% estimating mean_log_abs_diff slope  by linear regression
lreg_model = LinearRegression()
min_step = 1
max_step = 20
x = np.arange(min_step, max_step+1).reshape((-1, 1))


# linear regression to mean_log_abs_diff to obtain estimation of lyapunov exponent
lreg_model.fit(x, mean_log_abs_diff[min_step:max_step+1])  # %%
y_pred = lreg_model.predict(x)
corr_coef = np.corrcoef(mean_log_abs_diff[min_step:max_step+1], y_pred)[0, 1]
r2 = lreg_model.score(x, mean_log_abs_diff[min_step:max_step+1])

print("coefficient:", lreg_model.coef_[0])
print("seppen:", lreg_model.intercept_)
print("correlation coefficient:", corr_coef)
print("determination coefficient", r2)
# save to text file
with open(os.path.join(fig_dir, 'lyap_slope.txt'), mode='w') as f:
    f.write("slope: " + str(lreg_model.coef_[0]) + "\n")
    f.write("constant: " + str(lreg_model.intercept_) + "\n")
    f.write("correlation coefficient: " + str(corr_coef) + "\n")
    f.write("R^2: " + str(r2) + "\n")

# %%
# x0 = X_start[0].reshape(1,28,28,1)
# x0_p = x0 + 1e-5*tf.random.normal(x0.shape, dtype=np.float32) #type:ignore

# with tf.device('/cpu:0'):
#   x1 = generator_g(x0, training=False)
#   x1_1 = generator_g(x0, training=False)
#   x1_p = generator_g(x0_p, training=False)
#   x1_1_p = generator_g(x0_p, training=False)

# print(x1_1 - x1)
# print(x1_1_p - x1_p)

# d = x1 - x1_p
# d0 = x0 - x0_p
# print(d0)
# print(d)
# #%%


# print(np.sqrt(np.sum(d0*d0)))
# print(np.sqrt(np.sum(d*d)))

# %%
