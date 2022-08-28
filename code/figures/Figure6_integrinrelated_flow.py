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
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]


sns.kdeplot(sample["BB515-A"], ax = ax10, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)

sample = FCMeasurement(ID='ctrl', datafile=files[2])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

sns.kdeplot(sample["BB515-A"], ax = ax11, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)

############################
###########################

ax20 = fig.add_subplot(gs[1,0])
ax21 = fig.add_subplot(gs[1,1])

files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*Ctrl*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[7])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# ax20.hist(sample["BB515-A"], bins = 100, density = True,
#           lw = 2, color = '#B8BABC', histtype = 'step',)#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax20, color = 'k', ls = '-', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[6])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# _ = ax21.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#B8BABC')#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax21, color = 'k', ls = '-', lw = 1.5, legend = False)

############################
############################
ax30 = fig.add_subplot(gs[2,0])
ax31 = fig.add_subplot(gs[2,1])
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgITGB2*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[2])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]


# _ = ax30.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#738FC1')#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax30, color = '#738FC1', ls = '-', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[-1])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# _ = ax31.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#738FC1')#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax31, color = '#738FC1', ls = '-', lw = 1.5, legend = False)

############################
############################
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgVPS*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[3])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# _ = ax30.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#738FC1',
#              ls = '-.')#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax30, color = '#738FC1', ls = '-.', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[4])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

sns.kdeplot(sample["BB515-A"], ax = ax31, color = '#738FC1', ls = '-.', lw = 1.5, legend = False)

############################
############################
files = glob.glob('../../data/flow_cytometry/20220317_KDlines_integrinrelated_repeats/*sgSNX17*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[-1])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# _ = ax30.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#738FC1',
#              ls = '--')#, label  = samples[i])
sns.kdeplot(sample["BB515-A"], ax = ax30, color = '#738FC1', ls = '--', lw = 1.5, legend = False)

sample = FCMeasurement(ID='ctrl', datafile=files[3])
sample = sample.transform('hlog', channels=['FSC-A', 'SSC-A', 'BB515-A', 'FPR1: AF 647-A', 'BB700: PerCP-Cy5.5-A'], b=500.0)
sample = sample.data
sample = sample[sample['FSC-A'] >= 8.9E3]
sample = sample[sample['FSC-A'] <= 10E3]

# _ = ax31.hist(sample["BB515-A"], bins = 100, density = True,
#        histtype = 'step', lw = 2, color = '#738FC1',
#              ls = '--')#, label  = samples[i])

import seaborn as sns
sns.kdeplot(sample["BB515-A"], ax = ax31, color = '#738FC1', ls = '--', lw = 1.5, legend = False)

# ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# ax.set_xlabel('ITGB2 expression (fluorescence, a.u.)', fontsize = 20)
# ax[0].set_ylabel('frequency', fontsize = 20)

for ax_ in [ax10,ax11,ax20,ax21,ax30,ax31]:
    ax_.set_yticks([])
    ax_.set_xticks([0,2E3,4E3, 6E3, 8E3,10000])
    ax_.set_xlim(0,10000)
    ax_.grid(axis='x', color= 'grey', alpha  = 0.2)#xcolor='0.95')
#     ax_.set_xticklabels([])
#     ax_.axes.get_xaxis().set_visible(False)
    ax_.set_xticklabels([])


plt.tight_layout()
fig.savefig('../../figures/Figure6_integrins_flow.pdf',bbox_inches='tight')
