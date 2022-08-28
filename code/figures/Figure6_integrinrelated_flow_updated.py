import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

import seaborn as sns

plt.style.use('styleNB.mplstyle')


fig = plt.figure(figsize=(4,3))
gs = GridSpec(nrows=3, ncols=2,
              height_ratios=[1, 1, 1],
              width_ratios=[1, 1])


###################################
# flow  cytometry
################################################
############################
############################
ax10 = fig.add_subplot(gs[0,0])
ax11 = fig.add_subplot(gs[0,1])

files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*iso*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[1])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax10, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)

sample = FCMeasurement(ID='ctrl', datafile=files[2])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax11, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)

############################
###########################

ax20 = fig.add_subplot(gs[1,0])
ax21 = fig.add_subplot(gs[1,1])

files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*Ctrl*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[7])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax20, color = 'k', ls = '-', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[6])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax21, color = 'k', ls = '-', lw = 1.5, legend = False)

############################
############################
ax30 = fig.add_subplot(gs[2,0])
ax31 = fig.add_subplot(gs[2,1])
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgITGB2*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[2])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax30, color = '#738FC1', ls = '-', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[-1])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax31, color = '#738FC1', ls = '-', lw = 1.5, legend = False)

############################
############################
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgVPS*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[3])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax30, color = '#738FC1', ls = '-.', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[4])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax31, color = '#738FC1', ls = '-.', lw = 1.5, legend = False)

############################
############################
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgSNX17*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[-1])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax30, color = '#738FC1', ls = '--', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[3])
sample = sample.data

sample = sample[sample['FSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] >= 0.4E5]
sample = sample[sample['SSC-A'] <= 1.2E5]
# log values
sample['BB515-A_l'] = np.log10(sample['BB515-A'][sample['BB515-A'] > 0])

sns.kdeplot(sample["BB515-A_l"], ax = ax31, color = '#738FC1', ls = '--', lw = 1.5, legend = False)


for ax_ in [ax10,ax11,ax20,ax21,ax30,ax31]:
    ax_.set_yticks([])
    # ax_.set_xticks([0,2E3,4E3, 6E3, 8E3,10000])
    ax_.set_xlim(2,4.6)
    ax_.grid(axis='x', color= 'grey', alpha  = 0.2)#xcolor='0.95')

    # ax_.axes.get_xaxis().set_visible(False)
    ax_.axes.get_yaxis().set_visible(False)
    ax_.set_xlabel(None)
    ax_.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                            2000,3000,4000,5000,6000,7000,8000,9000,
                            20000,30000,40000,50000,60000,70000,80000,90000,
                            200000,300000]), minor = True)
    ax_.set_xticks(np.log10([100,1000, 10000,100000]))
    ax_.set_xticklabels([])
    ax_.tick_params(length=8)


plt.tight_layout()
fig.savefig('../../figures/Figure6_integrins_flow.pdf',bbox_inches='tight')
