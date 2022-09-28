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

import seaborn as sns

# plt.style.use('styleNB')
plt.style.use('styleNB.mplstyle')

# colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
# color = ['#738FC1', '#7AA974', '#CC462F', '#EAC264', '#97459B',
#          '#7CD6C4', '#D87E6A', '#BCDB8A', '#BF78C4', '#9653C1']

colors2 = sns.color_palette("Set2")
cell_lines_colors = {'sgCtrl1' : '#B8BABC',
  'sgFLCN': colors2[6],
  'sgLAMTOR1': colors2[0],
  'sgTSC1' :  '#738FC1',
  'sgRICTOR' :  '#738FC1'}
#############################################
#  Figure 7: Summary of RNA-seq results
#############################################


fig = plt.figure(figsize=(9,3))
gs = GridSpec(nrows=1, ncols=5, height_ratios=[1], width_ratios=[1, 1, 1, 1, 1])


ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])

#############################

###############################

# padj values
pw_dict = {'Neutrophil activation' : 1.12E-17,
 'Neutrophil degranulation' : 1.95E-16,
 'foal adhesion & cell-\nsubstrate junctions' : 3.24E-13,
 'Ficolin-1-rich granules' : 4.62E-13,
 'Ribosomes' : 7.56E-10,
 'Lysosome' : 4.57e-05,
 'Autophagy' : 5.277E-04,
 'mTOR signaling pathway' : 6.027E-02}

for i, val in enumerate(pw_dict):
    ax1.barh(len(pw_dict) - i -1.5, -np.log10(pw_dict[val]), color = colors2[6])
ax1.set_yticks([])
ax1.set_xlabel(r'-log$_{10}$(adjusted'
                '\np-value)')
ax1.invert_xaxis()
# ax1.spines.right.set_visible(True)
# ax1.spines.left.set_visible(False)
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(False)


updown_dict = {'Neutrophil activation' : [171, 188],
'Neutrophil degranulation' : [163, 181],
'foal adhesion & cell-\nsubstrate junctions' : [138, 141],
'Ficolin-1-rich granules' : [73, 70],
'Ribosomes' : [92, 5],
'Lysosome' : [64, 22],
'Autophagy' : [39, 46],
'mTOR signaling pathway' : [48, 50]}

count = 0
for i, val in enumerate(updown_dict):
    ax2.barh(len(updown_dict) - count, updown_dict[val][0], color = '#53B64C')
    ax2.barh(len(updown_dict) - count-1, updown_dict[val][1], color = '#E87D72')
    count += 2
ax2.set_yticks([])
ax2.set_xlabel('number of differentially\nexpressed genes')
ax2.set_yticks([])


plt.tight_layout()
fig.savefig('../../../figures/Figure3E_rna-seq_pwEnrichment.pdf', bbox_inches='tight')
