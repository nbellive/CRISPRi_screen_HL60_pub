import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import numpy as np
import glob
import pandas as pd
import math

import FlowCytometryTools
from FlowCytometryTools import FCMeasurement

import skimage
import skimage.io
from skimage.morphology import disk
from skimage.filters import rank
import scipy
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA


import seaborn as sns

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.05, 1.0), loc='upper left')

rg = np.random.default_rng()

def draw_bs_rep(data, func, rg):
    """Compute a bootstrap replicate from data."""
    bs_sample = rg.choice(data, size=len(data))
    return func(bs_sample)

plt.style.use('styleNB.mplstyle')
fig = plt.figure(figsize=(5,2))
gs = GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1])

# ax_B   = fig.add_subplot(gs[0,-2:])

ax_C1  = fig.add_subplot(gs[0])
ax_C2  = fig.add_subplot(gs[1])
ax_C3  = fig.add_subplot(gs[2])


ax_flow = [ax_C1, ax_C2, ax_C3]


files_ = ['../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_sgCtrl1_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_FLCN_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_LAMTOR1_Data Source - 1.fcs']

ax_ind = dict(zip(files_, np.arange(3)))

for i, f in enumerate(files_):
    sample = FCMeasurement(ID='ctrl', datafile=f)
    sample = sample.data

    # gate out the noise and dead cells
    sample = sample[sample['FSC-A'] >= 1.8E5]
    sample = sample[sample['FSC-A'] <= 4.1E5]
    sample = sample[sample['SSC-A'] >= 0.3E5]
    sample = sample[sample['SSC-A'] <= 1.5E5]
    sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])
    sample['AF 647-A_l'] = np.log10(sample['AF 647-A'][sample['AF 647-A']>0])

    g1 = sns.kdeplot(data=sample.iloc[:5000], x="BB515-A_l", y="AF 647-A_l",
                 shade=True, thresh=.2, cmap="rocket_r", ax = ax_flow[ax_ind[f]]) #cmap="Reds"
    g1.set(xlabel=None, ylabel=None)

for ax_ in ax_flow:
    ax_.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                            2000,3000,4000,5000,6000,7000,8000,9000,
                            20000,30000,40000,50000,60000,70000,80000,90000,
                            200000,300000]), minor = True)
    ax_.set_xticks(np.log10([100,1000, 10000,100000]))
    ax_.set_yticks(np.log10([70,80,90,200,300,400,500,600,700,800,900,
                            2000,3000]), minor = True)
    ax_.set_yticks(np.log10([100,1000]))
    ax_.set_xlim(2,5.5)
    ax_.set_ylim(1.8,3.5)
    for y in np.log10([100,1000]):
        ax_.hlines(y, 2,5.5,linewidth=1, alpha = 0.1, zorder=20)
    for x in np.log10([100,1000, 10000,100000]):
        ax_.vlines(x, 1.8,3.5,linewidth=1, alpha = 0.1, zorder=20)
    ax_.set_xticklabels([])
    ax_.set_yticklabels([])


plt.tight_layout()
fig.savefig('../../figures/FigS3_differentiation_flow.pdf')
