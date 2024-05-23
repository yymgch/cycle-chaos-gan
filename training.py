#######
# training cycleChaosGAN
# dataset and some parameters are defined in parameters.py
#######


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
os.makedirs(data_dir, exist_ok=True)

fig_dir_base = './figure'
fig_dir = os.path.join(fig_dir_base, exp_name, 'loss')
os.makedirs(fig_dir, exist_ok=True)
# %%

###### GPU setting #####

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

# BUFFER_SIZE = 1000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 20
IMG_WIDTH = 28  # not used
IMG_HEIGHT = 28
LAMBDA = 10  # weight for consistency loss

print(f'exp_name:{exp_name}')

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

set_random_seed()
# %%
##### loading MNIST or fashion-MNIST dataset #####
triplet_dataset = TripletDataset(
    DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
ds_train_x = triplet_dataset.ds_train_x
ds_train_y = triplet_dataset.ds_train_y
ds_train_z = triplet_dataset.ds_train_z

ds_test_x = triplet_dataset.ds_test_x
ds_test_y = triplet_dataset.ds_test_y
ds_test_z = triplet_dataset.ds_test_z

# %%
##### model initialization #####


generator_g = model.make_generator_model()
generator_f = model.make_generator_model()


discriminator_x = model.make_discriminator_model_addDropout()
discriminator_y = model.make_discriminator_model_addDropout()
discriminator_z = model.make_discriminator_model_addDropout()

# %%
##########
# input one mini-batch and check the shape of output

sample_x = next(iter(ds_train_x))  
_ = generator_g(sample_x)  # warm-up
print(_.shape)
_ = generator_f(sample_x)
print(_.shape)
_ = discriminator_x(sample_x)
print(_.shape)
_ = discriminator_y(sample_x)
print(_.shape)
_ = discriminator_z(sample_x)
print(_.shape)

# %%
# Loss functions and optimizers


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_z_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# %%
##### Checkpoints #####

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
# 
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print ('Latest checkpoint restored!!')

model.models_save_weights(checkpoint_path, 'weights-0', generator_g,
                          generator_f, discriminator_x, discriminator_y, discriminator_z)

# %%


def calc_loss(real_x, real_y, real_z, gen_g, gen_f):
    ''' calculate loss
    '''
    fake_y = gen_g(real_x, training=True)
    cycled_x = gen_f(fake_y, training=True)     

    fake_z = gen_g(real_y, training=True)
    cycled_y = gen_f(fake_z, training=True)

    fake_x = gen_g(real_z, training=True)
    cycled_z = gen_f(fake_x, training=True)

    fake_z_r = gen_f(real_x, training=True)
    cycled_x_r = gen_g(fake_z_r, training=True)

    fake_y_r = gen_f(real_z, training=True)
    cycled_z_r = gen_g(fake_y_r, training=True)

    fake_x_r = gen_f(real_y, training=True)
    cycled_y_r = gen_g(fake_x_r, training=True)


    # evaluations
    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)
    disc_real_z = discriminator_z(real_z, training=True)
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)
    disc_fake_z = discriminator_z(fake_z, training=True)
    disc_fake_x_r = discriminator_x(fake_x_r, training=True)
    disc_fake_y_r = discriminator_y(fake_y_r, training=True)
    disc_fake_z_r = discriminator_z(fake_z_r, training=True)

    # calculate the loss
    gen_g_loss_x = model.generator_loss(disc_fake_x)
    gen_g_loss_y = model.generator_loss(disc_fake_y)
    gen_g_loss_z = model.generator_loss(disc_fake_z)
    gen_g_loss = gen_g_loss_x + gen_g_loss_y + gen_g_loss_z
    gen_f_loss_x = model.generator_loss(disc_fake_x_r)
    gen_f_loss_y = model.generator_loss(disc_fake_y_r)
    gen_f_loss_z = model.generator_loss(disc_fake_z_r)
    gen_f_loss = gen_f_loss_x + gen_f_loss_y + gen_f_loss_z

    # cycle consistency loss
    xyx_cycle_loss = model.calc_cycle_loss(real_x, cycled_x, LAMBDA)
    yzy_cycle_loss = model.calc_cycle_loss(real_y, cycled_y, LAMBDA)
    zxz_cycle_loss = model.calc_cycle_loss(real_z, cycled_z, LAMBDA)
    xzx_cycle_loss = model.calc_cycle_loss(real_x, cycled_x_r, LAMBDA)
    yxy_cycle_loss = model.calc_cycle_loss(real_y, cycled_y_r, LAMBDA)
    zyz_cycle_loss = model.calc_cycle_loss(real_z, cycled_z_r, LAMBDA)
    # add all cycle losses
    total_cycle_loss = xyx_cycle_loss + yzy_cycle_loss + \
        zxz_cycle_loss + xzx_cycle_loss + yxy_cycle_loss + zyz_cycle_loss

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss
    total_gen_f_loss = gen_f_loss + total_cycle_loss

    disc_x_loss = model.discriminator_loss(
        disc_real_x, disc_fake_x, disc_fake_x_r)
    disc_y_loss = model.discriminator_loss(
        disc_real_y, disc_fake_y, disc_fake_y_r)
    disc_z_loss = model.discriminator_loss(
        disc_real_z, disc_fake_z, disc_fake_z_r)

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, disc_z_loss


@tf.function
def train_step(real_x, real_y, real_z, gen_g, gen_f):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.

  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, disc_z_loss = calc_loss(
        real_x, real_y, real_z, gen_g, gen_f)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(
      total_gen_g_loss, gen_g.trainable_variables)
  generator_f_gradients = tape.gradient(
      total_gen_f_loss, gen_f.trainable_variables)
  discriminator_x_gradients = tape.gradient(
      disc_x_loss, discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(
      disc_y_loss, discriminator_y.trainable_variables)
  discriminator_z_gradients = tape.gradient(
      disc_z_loss, discriminator_z.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(
      zip(generator_g_gradients, gen_g.trainable_variables))
  generator_f_optimizer.apply_gradients(
      zip(generator_f_gradients, gen_f.trainable_variables))
  discriminator_x_optimizer.apply_gradients(
      zip(discriminator_x_gradients, discriminator_x.trainable_variables))
  discriminator_y_optimizer.apply_gradients(
      zip(discriminator_y_gradients, discriminator_y.trainable_variables))
  discriminator_z_optimizer.apply_gradients(
      zip(discriminator_z_gradients, discriminator_z.trainable_variables))

@tf.function
def test_step(real_x, real_y, real_z, gen_g, gen_f):
  
  total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, disc_z_loss = calc_loss(
        real_x, real_y, real_z, gen_g, gen_f)

  return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, disc_z_loss


# run training
# 0->1
sample_x = next(iter(ds_test_x))
sample_y = next(iter(ds_test_y))
sample_z = next(iter(ds_test_z))


#%% prepare metrics for test-loss
test_loss_g = tf.keras.metrics.Mean(name='test_loss_g')
test_loss_f = tf.keras.metrics.Mean(name='test_loss_f')
test_loss_d_x = tf.keras.metrics.Mean(name='test_loss_d_x')
test_loss_d_y = tf.keras.metrics.Mean(name='test_loss_d_y')
test_loss_d_z = tf.keras.metrics.Mean(name='test_loss_d_z')

# array for recording loss
test_losses_g = []
test_losses_f = []
test_losses_d_x = []
test_losses_d_y = []
test_losses_d_z = []

# %%
##### main training loop #####

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y, image_z in tf.data.Dataset.zip((ds_train_x, ds_train_y, ds_train_z)):
    train_step(image_x, image_y, image_z, generator_g, generator_f)
    if n % 10 == 0:
      print('.', end='')
    n += 1

  # clear_output(wait=True)
  
    #%% test loss 
  # reset the metrics at the start of the next epoch
  test_loss_g.reset_states()
  test_loss_f.reset_states()
  test_loss_d_x.reset_states()
  test_loss_d_y.reset_states()
  test_loss_d_z.reset_states()

  for image_x, image_y, image_z in tf.data.Dataset.zip((ds_test_x, ds_test_y, ds_test_z)):
    gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, disc_z_loss = test_step(image_x, image_y, image_z, generator_g, generator_f)
    
    test_loss_g(gen_g_loss)
    test_loss_f(gen_f_loss)    
    test_loss_d_x(disc_x_loss)
    test_loss_d_y(disc_y_loss)
    test_loss_d_z(disc_z_loss)
  # record loss
  test_losses_g.append(test_loss_g.result())
  test_losses_f.append(test_loss_f.result())
  test_losses_d_x.append(test_loss_d_x.result())
  test_losses_d_y.append(test_loss_d_y.result())
  test_losses_d_z.append(test_loss_d_z.result())
  print(f'Epoch {epoch+1}, Test Gen G Loss: {test_loss_g.result()}, Test Gen F Loss: {test_loss_f.result()}, Test Disc X Loss: {test_loss_d_x.result()}, Test Disc Y Loss: {test_loss_d_y.result()}, Test Disc Z Loss: {test_loss_d_z.result()}') 


  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                        ckpt_save_path))

  print('Time taken for epoch {} is {} sec\n'.format(
      epoch + 1, time.time()-start))

#%%
# plot loss

fig, ax = plt.subplots()
ax.plot(test_losses_g, label='Gen G')
ax.plot(test_losses_f, label='Gen F')
ax.plot(test_losses_d_x, label='Disc X')
ax.plot(test_losses_d_y, label='Disc Y')
ax.plot(test_losses_d_z, label='Disc Z')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig(os.path.join(fig_dir, 'loss.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'loss.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'loss.pdf'))

# %% save loss

np.save(os.path.join(data_dir, 'test_losses_g.npy'), test_losses_g)
np.save(os.path.join(data_dir, 'test_losses_f.npy'), test_losses_f)
np.save(os.path.join(data_dir, 'test_losses_d_x.npy'), test_losses_d_x)
np.save(os.path.join(data_dir, 'test_losses_d_y.npy'), test_losses_d_y)
np.save(os.path.join(data_dir, 'test_losses_d_z.npy'), test_losses_d_z) 


# %%
