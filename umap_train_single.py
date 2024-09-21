#%%

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

from util import TripletDataset, generate_images
from parameters import exp_name, name_dataset, DIGITS_PAIR, set_random_seed

set_random_seed() 
data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'umap_train')
os.makedirs(data_dir, exist_ok=True)

fig_base_dir = './figures'
fig_dir = os.path.join(fig_base_dir, exp_name, 'umap_train')
os.makedirs(fig_dir, exist_ok=True)

#%%

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

AUTOTUNE = tf.data.AUTOTUNE

tf.config.experimental.enable_tensor_float_32_execution(
    False)  # tf32 (混合精度)を使わない
tf.keras.backend.set_floatx('float32')
mydtype = tf.float32
mynpdtype = np.float32

BUFFER_SIZE = 10000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

bs = BATCH_SIZE
tbs = TEST_BATCH_SIZE

#%%
##### training and test dataset #####

# loading dataset
triplet_dataset = TripletDataset(DIGITS_PAIR, bs, tbs, name_dataset=name_dataset)
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


train_data = np.concatenate([train_images_x, train_images_y, train_images_z]).reshape((-1,28*28))

#%%

mapper = UMAP(n_neighbors=15, min_dist=0.1, random_state=42) # fix random state # n_neighbors=15 and min_dist=0.1 is default value 
mapper.fit(train_data)
u_train= mapper.transform(train_data) # type:ignore

print(np.shape(u_train)) # type:ignore
# (18831, 2)

print(len(train_images_x) + len(train_images_y) +len(train_images_z))
# 18831

#%%
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

#%%
fig = plt.figure()
last0 = len(train_images_x) 
last1 = len(train_images_x) + len(train_images_y) 
last2 = len(train_images_x) + len(train_images_y) + len(train_images_z) 
fname  = 'umap_' + name_dataset + '_train'
plt.scatter(u_train[0 : last0, 0], u_train[0 : last0, 1], s=0.15, c = "lightgreen") # 主成分をプロット 0 #type:ignore
plt.scatter(u_train[last0  : last1, 0], u_train[last0  : last1, 1], s=0.15, c = "pink") # 主成分をプロット 1 # type:ignore
plt.scatter(u_train[last1  : last2, 0], u_train[last1  : last2, 1], s=0.15, c = "cyan") # 主成分をプロット 2 # type:ignore
plt.grid()
fig.savefig(os.path.join(fig_dir, fname+'.png'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight',pad_inches = 0.1)
#%% check misclustering

amb01 = train_images_x[u_train[0:last0,0]>10] # type: ignore
amb21 = train_images_z[u_train[last1:last2,0]>10]# type:ignore

#%%
plt.imshow(amb21[9,:,:,0], cmap='gray')

#%%
# 訓練データの2次元表現 u_train に対しk-means clustering をk=3でおこなう

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, rand_score
kmeans = KMeans(n_clusters=3, random_state=0).fit(u_train)
# 訓練データを分類した結果を得る

labels_umap = kmeans.labels_

# 正解のラベルを作る
labels = np.zeros(len(train_images_x) + len(train_images_y) + len(train_images_z))
labels[len(train_images_x):len(train_images_x)+len(train_images_y)] = 1
labels[len(train_images_x)+len(train_images_y):] = 2

# 正解ラベルと分類結果の一致度を計算
ars = adjusted_rand_score(labels, labels_umap)
print(f'Adjusted Rand Score : {ars}')

rand_s = rand_score(labels, labels_umap)
print(f'Rand Score : {rand_s}')

# ARIとRI をテキスト形式で保存
with open(os.path.join(data_dir, 'ARI_RI.txt'), 'w') as f:
    f.write(f'Adjusted Rand Score : {ars}\n')
    f.write(f'Rand Score : {rand_s}\n')

plt.hist(labels_umap, bins=3)

fig = plt.figure()
fname  = 'umap_' + name_dataset + '_train'
plt.scatter(u_train[labels_umap==0, 0], u_train[labels_umap==0, 1], s=0.15, c = "lightgreen") # 主成分をプロット 0 #type:ignore
plt.scatter(u_train[labels_umap==1, 0], u_train[labels_umap==1, 1], s=0.15, c = "pink") # 主成分をプロット 1 # type:ignore
plt.scatter(u_train[labels_umap==2, 0], u_train[labels_umap==2, 1], s=0.15, c = "cyan") 



#%% loading models

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


#checkpoints
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
    print ('Latest checkpoint restored!!')

#%%
    
# model.models_save_weights(checkpoint_path, 'weights-0', generator_g, generator_f, discriminator_x, discriminator_y, discriminator_z)


# fig_dir='./figures/training_figs'
# os.makedirs(fig_dir, exist_ok=True)
# fig = plt.figure()

#  (generator_g or generator_f)
gen_g = generator_g

test_input = next(iter(ds_test_x))
print(test_input.shape) #type:ignore

iteration = 5000


generated_y0 = gen_g(test_input)
generated_z0 = gen_g(generated_y0)
generated_x1 = gen_g(generated_z0)

generated_y = generated_y0.numpy()
generated_z = generated_z0.numpy()
generated_x = generated_x1.numpy()

#%%

im_seq = [test_input]
x = test_input

print('generate image sequences from test images')
for j in tqdm(range(iteration)):
    x_new = gen_g(x)
    im_seq.append(x_new)
    x = x_new
    # if( (j+1) %100 ==0):
    #     print(j+1, end=' ')

im_seq = np.array(im_seq).transpose(1,0,2,3,4) # batch, time, h, w, c
print(im_seq.shape)
#%%
    
ux = mapper.transform(im_seq.reshape(-1, 28*28))
print(ux.shape) # type:ignore (bs*(loop*3+1_, 2)

ux = np.array(ux).reshape((bs, iteration+1, 2    ))
#%%
fig, ax = plt.subplots()
ax.grid()
ax.scatter(u_train[0 : last0, 0], u_train[0 : last0, 1], s=0.15, c = "lightgreen") # 0 #type:ignore
ax.scatter(u_train[last0  : last1, 0], u_train[last0  : last1, 1], s=0.15, c = "pink") # 1 # type:ignore
ax.scatter(u_train[last1  : last2, 0], u_train[last1  : last2, 1], s=0.15, c = "cyan") # 2 # type:ignore

j = 0
fname = 'umap_' + name_dataset + '_train_single5000'
# ax.scatter(ux[j, 0:initial, 0], ux[j, 0:initial, 1], s=0.2, c='black') # type.ignore
ax.scatter(ux[j, :, 0], ux[j, :, 1], s=0.2, c='purple') # type.ignore
fig.savefig(os.path.join(fig_dir, fname+'.png'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight',pad_inches = 0.1)



# %% 64のsubplot from different initial points
# omit labels and ticks label

initial = 1000
fig, ax = plt.subplots(8,8, figsize=(20, 20))
ax = ax.flatten()
for j in range(64):
    ax[j].grid()
    ax[j].scatter(u_train[0 : last0, 0], u_train[0 : last0, 1], s=0.15, c = "lightgreen") # 主成分をプロット 0 #type:ignore
    ax[j].scatter(u_train[last0  : last1, 0], u_train[last0  : last1, 1], s=0.15, c = "pink") # 主成分をプロット 1 # type:ignore
    ax[j].scatter(u_train[last1  : last2, 0], u_train[last1  : last2, 1], s=0.15, c = "cyan") # 主成分をプロット 2 # type:ignore
    # ax[j].scatter(ux[j, 0:initial, 0], ux[j, 0:initial, 1], s=0.2, c='black') # type.ignore
    ax[j].scatter(ux[j, initial:, 0], ux[j, initial:, 1], s=0.2, c='purple') # type.ignore
    ax[j].set_xticks([])
    ax[j].set_yticks([])
fig.tight_layout()
fname = 'umap_' + name_dataset + '_train_64ics'
fig.savefig(os.path.join(fig_dir, fname+'.png'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight',pad_inches = 0.1)




#%%　trustworthiness and continuity を使ってuxのembeddingを評価する

from sklearn.manifold import trustworthiness

n_trj = 10
x_orig = im_seq[0:n_trj].reshape((-1,28*28))
x_emb = ux[0:n_trj,:,:].reshape((-1,2))
trw = trustworthiness(x_orig, x_emb, n_neighbors=5)
print(f'trustworthiness: {trw}')

continuity = trustworthiness(x_emb, x_orig, n_neighbors=5)

print(f'continuity: {continuity}')

# 結果をテキストファイルで保存

with open(os.path.join(data_dir, 'trustworthiness_continuity.txt'), 'w') as f:
    f.write(f'Trustworthiness : {trw}\n')
    f.write(f'continuity : {continuity}\n')


# %%
# same for PCA

# %%
from sklearn.decomposition import PCA
mapper = PCA(n_components=2)
mapper.fit(train_data)
u_train= mapper.transform(train_data) # type:ignore
# sequence data 
ux = mapper.transform(im_seq.reshape(-1, 28*28))    # type:ignore
ux = np.array(ux).reshape((bs, iteration+1, 2    ))
#%% plot
fig, ax = plt.subplots()
ax.grid()
ax.scatter(u_train[0 : last0, 0], u_train[0 : last0, 1], s=0.15, c = "lightgreen") #  0 #type:ignore
ax.scatter(u_train[last0  : last1, 0], u_train[last0  : last1, 1], s=0.15, c = "pink") #  1 # type:ignore
ax.scatter(u_train[last1  : last2, 0], u_train[last1  : last2, 1], s=0.15, c = "cyan") #  2 # type:ignore

j = 0
ax.scatter(ux[j, :, 0], ux[j, :, 1], s=0.2, c='purple') # type.ignore
fname = 'pca_' + name_dataset + '_train_single5000'
fig.savefig(os.path.join(fig_dir, fname+'.png'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight',pad_inches = 0.1)


# %%

fig, ax = plt.subplots(8,8, figsize=(20, 20))
ax = ax.flatten()
for j in range(64):
    ax[j].grid()
    ax[j].scatter(u_train[0 : last0, 0], u_train[0 : last0, 1], s=0.15, c = "lightgreen") #  0 #type:ignore
    ax[j].scatter(u_train[last0  : last1, 0], u_train[last0  : last1, 1], s=0.15, c = "pink") #  1 # type:ignore
    ax[j].scatter(u_train[last1  : last2, 0], u_train[last1  : last2, 1], s=0.15, c = "cyan") # 2 # type:ignore
    # ax[j].scatter(ux[j, 0:initial, 0], ux[j, 0:initial, 1], s=0.2, c='black') # type.ignore
    ax[j].scatter(ux[j, initial:, 0], ux[j, initial:, 1], s=0.2, c='purple') # type.ignore
    ax[j].set_xticks([])
    ax[j].set_yticks([])
fig.tight_layout()
fname = 'pca_' + name_dataset + '_train_64ics'
fig.savefig(os.path.join(fig_dir, fname+'.png'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.tiff'), bbox_inches='tight',pad_inches = 0.1, dpi=360)
fig.savefig(os.path.join(fig_dir, fname+'.pdf'), bbox_inches='tight',pad_inches = 0.1)

# %%saving


Xs = im_seq
np.save(os.path.join(data_dir, 'Xs.npy'), Xs)
np.save(os.path.join(data_dir, 'ux.npy'), ux)
np.save(os.path.join(data_dir, 'u_train.npy'), u_train)

