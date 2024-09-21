# %%
# Umap visualization of the set of trajectory

# here additional analysis for revicing manuscript is performed
# augmentation of initial condition by perturbation is applied and more number of initial conditions are used for analysis



from sklearn.decomposition import PCA
from parameters import exp_name, name_dataset, DIGITS_PAIR, set_random_seed
from util import TripletDataset, generate_images
import tensorflow as tf
import numpy as np
import os
import model
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from tensorflow.keras import layers
from re import A
from umap import UMAP
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


set_random_seed()  # fix the seeds  of random,numpy, tensorflow
data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'umap_train_many')
os.makedirs(data_dir, exist_ok=True)

fig_base_dir = './figures'
fig_dir = os.path.join(fig_base_dir, exp_name, 'umap_train_many')
os.makedirs(fig_dir, exist_ok=True)

# %%

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
    False)  
tf.keras.backend.set_floatx('float32')
mydtype = tf.float32
mynpdtype = np.float32

BUFFER_SIZE = 10000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

# %%
##### training dataset #####

# loading
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z


train_images_x = triplet_dataset.train_images_x
train_images_y = triplet_dataset.train_images_y
train_images_z = triplet_dataset.train_images_z

test_images_x = triplet_dataset.test_images_x
test_images_y = triplet_dataset.test_images_y
test_images_z = triplet_dataset.test_images_z


train_data = np.concatenate(
    [train_images_x, train_images_y, train_images_z]).reshape((-1, 28*28))

# %%

# fix random state # n_neighbors=15 and min_dist=0.1 is default value
mapper = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
mapper.fit(train_data)
u_train = mapper.transform(train_data)  # type:ignore

print(np.shape(u_train))  # type:ignore
# (18831, 2)

print(len(train_images_x) + len(train_images_y) + len(train_images_z))
# 18831

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

# %%　plot only training data
fig = plt.figure()
last0 = len(train_images_x)
last1 = len(train_images_x) + len(train_images_y)
last2 = len(train_images_x) + len(train_images_y) + len(train_images_z)
fname = 'umap_' + name_dataset + '_train'
plt.scatter(u_train[0: last0, 0], u_train[0: last0, 1],  # type:ignore
            s=0.15, c="lightgreen")  # #type:ignore
plt.scatter(u_train[last0: last1, 0], u_train[last0: last1,  # type:ignore
            1], s=0.15, c="pink")  # 1 # type:ignore
plt.scatter(u_train[last1: last2, 0], u_train[last1: last2,  # type:ignore
            1], s=0.15, c="cyan")  #  2 # type:ignore
plt.grid()
fig.savefig(os.path.join(fig_dir, fname+'.png'),
            bbox_inches='tight', pad_inches=0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'),
            bbox_inches='tight', pad_inches=0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'),
            bbox_inches='tight', pad_inches=0.1)
# %% check misclustering

amb01 = train_images_x[u_train[0:last0, 0] > 10]  # type: ignore
amb21 = train_images_z[u_train[last1:last2, 0] > 10]  # type:ignore

# %%
plt.imshow(amb21[9, :, :, 0], cmap='gray')


# %% loading models

generator_g = model.make_generator_model()
generator_f = model.make_generator_model()


discriminator_x = model.make_discriminator_model_addDropout()
discriminator_y = model.make_discriminator_model_addDropout()
discriminator_z = model.make_discriminator_model_addDropout()

# check
# 0->1
im = next(iter(ds_train_x))
g_x = generator_g(im)
print(f'generator output shape:{g_x.shape}')
d = discriminator_y(g_x)
print(f'discriminator output shape:{d.shape}')
fg_x = generator_f(g_x)
d_y = discriminator_x(fg_x)

# fig = model.generate_images(generator_g, generator_f, im)
# plt.show()

# 1->0
im = next(iter(ds_train_y))
f_y = generator_f(im)
print(f'generator output shape:{f_y.shape}')
d = discriminator_x(f_y)
print(f'discriminator output shape:{d.shape}')
gf_y = generator_g(f_y)
d_x = discriminator_y(gf_y)

# fig = model.generate_images2(generator_f, generator_g, im)
# plt.show()


# 2->0
im = next(iter(ds_train_z))
g_z = generator_g(im)
print(f'generator output shape:{g_z.shape}')
d = discriminator_y(g_z)
print(f'discriminator output shape:{d.shape}')
fg_z = generator_f(g_z)
d_x = discriminator_z(fg_z)

# fig = model.generate_images(generator_g, generator_f, im)
# plt.show()


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

# %%

# model.models_save_weights(checkpoint_path, 'weights-0', generator_g, generator_f, discriminator_x, discriminator_y, discriminator_z)


# %% testing
# fig_dir='./figures/training_figs'
# os.makedirs(fig_dir, exist_ok=True)
# fig = plt.figure()

# set generator (generator_g or generator_f)
gen_g = tf.function(generator_g)
# 入力画像
test_input = next(iter(ds_test_x))
print(test_input.shape)  # type:ignore
# set iteration number
iteration = 5000

# set initial value to test image 
generated_y0 = gen_g(test_input)
# 
generated_z0 = gen_g(generated_y0)
# 
generated_x1 = gen_g(generated_z0)

generated_y = generated_y0.numpy() #type:ignore
generated_z = generated_z0.numpy() # type:ignore
generated_x = generated_x1.numpy() #type:ignore

# %% generation
# steps to be used for analysis
steps_used = np.concatenate(
    [np.arange(0, 10, dtype=int),
     np.arange(10, 100, 10, dtype=int),
     np.arange(100, 1000, 100, dtype=int),
     np.arange(1000, iteration+1, 1000, dtype=int)])
print(steps_used)
# %% prepare perturbation

n_perturb = 5 # number of perturbation per one image

X_0 = triplet_dataset.test_images_x # un-perturbed images (980,28,28,1)
X_0 = X_0.reshape(X_0.shape[0], -1) # (980, 28*28)  
#calc distance between images
sqX_0 = np.sum(X_0**2, axis=1)

D = np.expand_dims(sqX_0, 0) + np.expand_dims(sqX_0, 1) - 2 * np.matmul(X_0, X_0.T) # distance matrix
np.fill_diagonal(D, np.inf) # excluding diagonal elements
minD = np.min(D, axis=1)
arg_minD = np.argmin(D, axis=1) # 最近傍のindex

X_0_nearest = X_0[arg_minD] # 最近傍の画像

ds_X0X0nearest = tf.data.Dataset.from_tensor_slices((X_0, X_0_nearest)).batch(50)

#%%
plt.imshow(X_0[-2,:].reshape((28,28)))
plt.show()
plt.imshow(X_0_nearest[-2,:].reshape((28,28)))
D.shape

def perturb(x0, x0_nearest, epsilon):
  x = x0 + np.random.randn(*x0.shape) * epsilon*(x0_nearest - x0)
  return x
#%%

Xss = []


for x0, x0_nearest in ds_X0X0nearest:
  x = x0
  for k in range(n_perturb):
    x = tf.reshape(perturb(x0, x0_nearest, 0.1),(-1, 28, 28, 1))
    im_seq = [x]
    for j in tqdm(range(iteration)):
      x_new = gen_g(x)
      im_seq.append(x_new)
      x = x_new
    xs = np.array(im_seq).transpose(1, 0, 2, 3, 4)
    Xss.append(xs[:,steps_used])
    # Xss.append(np.array(im_seq).transpose(1, 0, 2, 3, 4))  # batch, time, h,w,c
# %% saving images
print('generate image sequences from test images')
Xs = np.concatenate(Xss, axis=0) # (n_sample, n_step_used, 28, 28, 1)
Xs_used = Xs.reshape(Xs.shape[0], Xs.shape[1], -1)
n_sample = Xs.shape[0]
n_step = len(steps_used)
print(Xs.shape)
print(Xs_used.shape)
# %% saving Xs, Xs_used, step_used 
np.save(os.path.join(data_dir, 'Xs_perturb.npy'), Xs)
np.save(os.path.join(data_dir, 'Xs_used_perturb.npy'), Xs_used)
np.save(os.path.join(data_dir, 'steps_used_perturb.npy'), steps_used)


# %%
# map trajectory
ux = mapper.transform(Xs_used.reshape(-1, 28*28))
print(ux.shape)  # type:ignore (bs*(loop*3+1_, 2)

ux = np.array(ux).reshape((n_sample, n_step, 2))
# %%　

for j in range(n_step):
  step = steps_used[j]
  ux_step = ux[:, j, :]

  fig, ax = plt.subplots()
  ax.grid()
  ax.scatter(u_train[0: last0, 0], u_train[0: last0, 1],  # type:ignore
             s=0.15, c="lightgreen")  #  0 #type:ignore
  ax.scatter(u_train[last0: last1, 0], u_train[last0: last1,  # type:ignore
             1], s=0.15, c="pink")  #  1 # type:ignore
  ax.scatter(u_train[last1: last2, 0], u_train[last1: last2,  # type:ignore
             1], s=0.15, c="cyan")  #  2 # type:ignore
  # step number
  ax.text(0.05, 0.85, f'n={step}', transform=ax.transAxes, fontsize=22)

  fname = 'umap_' + name_dataset + '_many_step_perturb' + str(step)

  ax.scatter(ux_step[:, 0], ux_step[:, 1], s=0.2, c='purple')  # type.ignore
  fig.savefig(os.path.join(fig_dir, fname+'.png'),
              bbox_inches='tight', pad_inches=0.1, dpi=360)
  fig.savefig(os.path.join(fig_dir, fname+'.tiff'),
              bbox_inches='tight', pad_inches=0.1, dpi=360)
  fig.savefig(os.path.join(fig_dir, fname+'.pdf'),
              bbox_inches='tight', pad_inches=0.1)


# %%  perturbation for y and z in the same way to x

Y_0 = triplet_dataset.test_images_y # un-perturbed images (n_y,28,28,1)
Y_0 = Y_0.reshape(Y_0.shape[0], -1) # (980, 28*28)  
#calc distance between images
sqY_0 = np.sum(Y_0**2, axis=1)

D_Y = np.expand_dims(sqY_0, 0) + np.expand_dims(sqY_0, 1) - 2 * np.matmul(Y_0, Y_0.T) # distance matrix
np.fill_diagonal(D_Y, np.inf) # excluding diagonal elements
minD = np.min(D_Y, axis=1)
arg_minD = np.argmin(D_Y, axis=1) # 最近傍のindex

Y_0_nearest = Y_0[arg_minD] # 最近傍の画像

ds_Y0Y0nearest = tf.data.Dataset.from_tensor_slices((Y_0, Y_0_nearest)).batch(50)

#%%
Xss_y = []


#%%

for y0, y0_nearest in ds_Y0Y0nearest:
  x = x0
  for k in range(n_perturb):
    x = tf.reshape(perturb(y0, y0_nearest, 0.1),(-1, 28, 28, 1))
    im_seq = [x]
    for j in tqdm(range(iteration)):
      x_new = gen_g(x)
      im_seq.append(x_new)
      x = x_new
    xs = np.array(im_seq).transpose(1, 0, 2, 3, 4)
    Xss_y.append(xs[:,steps_used])

# %%
Xs_y = np.concatenate(Xss_y, axis=0) # (n_sample, n_step_used, 28, 28, 1)
Xs_used_y = Xs_y.reshape(Xs_y.shape[0], Xs_y.shape[1], -1)
n_sample_y = Xs_y.shape[0]
n_step = len(steps_used)
print(Xs_y.shape)
print(Xs_used_y.shape)

# %% Xs_y, Xs_used_y を保存
np.save(os.path.join(data_dir, 'Xs_y_perturb.npy'), Xs_y)
np.save(os.path.join(data_dir, 'Xs_used_y_perturb.npy'), Xs_used_y)
# %%

# map
ux_y = mapper.transform(Xs_used_y.reshape(-1, 28*28))
print(ux_y.shape)  # type:ignore

ux_y = np.array(ux_y).reshape((n_sample_y, n_step, 2))
# %% same for z

Z_0 = triplet_dataset.test_images_z # un-perturbed images (n_z,28,28,1)
Z_0 = Z_0.reshape(Z_0.shape[0], -1) # (980, 28*28)  
#calc distance between images
sqZ_0 = np.sum(Z_0**2, axis=1)

D_Z = np.expand_dims(sqZ_0, 0) + np.expand_dims(sqZ_0, 1) - 2 * np.matmul(Z_0, Z_0.T) # distance matrix
np.fill_diagonal(D_Z, np.inf) # excluding diagonal elements
minD = np.min(D_Z, axis=1)
arg_minD = np.argmin(D_Z, axis=1) # 最近傍のindex

Z_0_nearest = Z_0[arg_minD] # 最近傍の画像

ds_Z0Z0nearest = tf.data.Dataset.from_tensor_slices((Z_0, Z_0_nearest)).batch(50)


Xss_z = []
#%%


for z0, z0_nearest in ds_Z0Z0nearest:
  x = x0
  for k in range(n_perturb):
    x = tf.reshape(perturb(z0, z0_nearest, 0.1),(-1, 28, 28, 1))
    im_seq = [x]
    for j in tqdm(range(iteration)):
      x_new = gen_g(x)
      im_seq.append(x_new)
      x = x_new
    xs = np.array(im_seq).transpose(1, 0, 2, 3, 4)
    Xss_z.append(xs[:,steps_used])

# %%
Xs_z = np.concatenate(Xss_z, axis=0) # (n_sample, n_step_used, 28, 28, 1)
Xs_used_z = Xs_z.reshape(Xs_z.shape[0], Xs_z.shape[1], -1)

n_sample_z = Xs_z.shape[0]
n_step = len(steps_used)
print(Xs_z.shape)
print(Xs_used_z.shape)
# %% save Xs_y, Xs_used_y 
np.save(os.path.join(data_dir, 'Xs_z_perturb.npy'), Xs_z)
np.save(os.path.join(data_dir, 'Xs_used_z_perturb.npy'), Xs_used_z)
# %%

# map
ux_z = mapper.transform(Xs_used_z.reshape(-1, 28*28))
print(ux_z.shape)  # type:ignore

ux_z = np.array(ux_z).reshape((n_sample_z, n_step, 2))


# %% ux, ux_y, ux_z 


for j in range(n_step):
  step = steps_used[j]
  ux_step = ux[:, j, :]
  ux_y_step = ux_y[:, j, :]
  ux_z_step = ux_z[:, j, :]

  fig, ax = plt.subplots()
  ax.grid()
  ax.scatter(u_train[0: last0, 0], u_train[0: last0, 1],  # type:ignore
             s=0.15, c="lightgreen")  #  0 #type:ignore
  ax.scatter(u_train[last0: last1, 0], u_train[last0: last1,  # type:ignore
             1], s=0.15, c="pink")  # 1 # type:ignore
  ax.scatter(u_train[last1: last2, 0], u_train[last1: last2,  # type:ignore
             1], s=0.15, c="cyan")  # 2 # type:ignore
  # display step number
  ax.text(0.05, 0.85, f'n={step}', transform=ax.transAxes, fontsize=22)

  fname = 'umap_' + name_dataset + 'xyz_many_step_perturb' + str(step)

  ax.scatter(ux_step[:, 0], ux_step[:, 1], s=0.2, c='darkgreen')  # type.ignore
  ax.scatter(ux_y_step[:, 0], ux_y_step[:, 1],
             s=0.2, c='darkred')  # type.ignore
  ax.scatter(ux_z_step[:, 0], ux_z_step[:, 1],
             s=0.2, c='darkblue')  # type.ignore
  fig.savefig(os.path.join(fig_dir, fname+'.png'),
              bbox_inches='tight', pad_inches=0.1, dpi=360)
  fig.savefig(os.path.join(fig_dir, fname+'.tiff'),
              bbox_inches='tight', pad_inches=0.1, dpi=360)
  fig.savefig(os.path.join(fig_dir, fname+'.pdf'),
              bbox_inches='tight', pad_inches=0.1)


