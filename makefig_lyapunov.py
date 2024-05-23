# %% making figures for Lyapunov spectrum analysis
import model
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
from re import A

from parameters import exp_name, name_dataset, DIGITS_PAIR
# matplotlibの設定
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

data_dir_base = './data'
data_dir = os.path.join(data_dir_base, exp_name, 'lyap')

fig_dir = './figures/' + exp_name + '/lyap'
os.makedirs(fig_dir, exist_ok=True)
# %%

Xs = np.load(os.path.join(data_dir_base, exp_name, 'lyap', 'Xs.npy'))
print(Xs.shape)  # (980, t_len, 784)

# %%
logrs = np.load(os.path.join(data_dir_base, exp_name, 'lyap', 'logrs.npy'))
lsps = np.load(os.path.join(data_dir_base, exp_name, 'lyap', 'lsps.npy'))


# %%
# example
plt.grid()
plt.plot(lsps[0])

# %%
# first 10
plt.grid()
plt.plot(lsps[0, 0:10])
for i in range(11):
    print(str(i) + ":" + str(lsps[0, i]))

# %%
# all 980
plt.grid()
for i in range(len(lsps)):
    plt.plot(lsps[i])

# %%
# all trajectories
fig, ax = plt.subplots()
ax.grid()
for i in range(len(lsps)):
    ax.plot(lsps[i, 0:10])
fig.savefig(os.path.join(fig_dir, 'lyap_0_10_10samples.png'))

# %%
# histgram of the first 5 Lyapunov exponents

fig, ax = plt.subplots()
ax.grid()
for i in range(5):
    ax.hist(lsps[:, i], bins=100, alpha=0.8,
            label='$\lambda_{'+str(i+1)+'}$')  # type:ignore
ax.legend(bbox_to_anchor=(0.94, 0.99), loc='upper left')
# show only left and bottom frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'lyap_0_10_hist.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'lyap_0_10_hist.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'lyap_0_10_hist.pdf'), )
# %%
# mean of 980  trajectories
enlarge_until = 15

lsps_mean = np.mean(lsps, axis=0)
# ignore inf (rare case)
lsps_nan = lsps.copy()
lsps_nan[np.isinf(lsps)] = np.nan

lsps_mean = np.nanmean(lsps_nan, axis=0)
# %%


fig, ax = plt.subplots()
ax.grid()
ax.plot(1+np.arange(len(lsps_mean)), lsps_mean)
ax.set_xlabel('$i$')
ax.set_ylabel('$\lambda_{i}$')  # type:ignore
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# enlarged subfigure in the lower left corner of the figure

axins = ax.inset_axes([0.16, 0.1, 0.45, 0.45])
axins.grid()
axins.plot(
    1+np.arange(len(lsps_mean[:enlarge_until])), lsps_mean[:enlarge_until], 'o-')
axins.set_xlim(0, enlarge_until)
axins.set_ylim(-0.5, 0.7)
axins.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'lyaps_mean.tiff'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'lyaps_mean.png'), dpi=360)
fig.savefig(os.path.join(fig_dir, 'lyaps_mean.pdf'))

# %% save text file
np.savetxt(os.path.join(data_dir, 'lyaps_mean.txt'), lsps_mean)

print(f'The largest Lyapunov exponent is {lsps_mean[0]}')
#######
# %% Lyapunov Dimension
#######


l_cumsum = np.cumsum(lsps_mean)
print(l_cumsum[0:20])
k = np.argmax(l_cumsum > 0)
# find maximum of k that satisfy l_cumsum[k]>0
k = np.where(l_cumsum > 0)[0][-1]

lyapunov_dim = 1+k + l_cumsum[k]/abs(lsps_mean[k+1])
print(f'Lyapunov dimension is {lyapunov_dim}')

# %% save text file
np.savetxt(os.path.join(data_dir, 'lyapunov_dim.txt'),
           np.array([lyapunov_dim]))


# %%
