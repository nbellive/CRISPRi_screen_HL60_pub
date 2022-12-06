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

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']

colors2 = sns.color_palette("Set2")

cell_lines_colors = {'sgControl' : '#B8BABC',
  'sgCtrl1' : '#B8BABC',
  'sgFLCN': colors2[3],
  'sgLAMTOR1':   sns.color_palette("husl", 8)[6],
  'sgSPI1' :  colors2[0],
  'sgTSC1' :  '#738FC1',
  'sgRICTOR' : '#738FC1',
  'sgCEBPE' : '#738FC1'}

colors = ['#E28027', '#078ED1', 'grey']
#############################################
#  Figure S1 - comparison of different dCas9-KRAB constructs in uHL-60 cells
#############################################

fig = plt.figure(figsize=(6,1.75))
gs = GridSpec(nrows=1, ncols=4)

ax0  = fig.add_subplot(gs[0])
ax1  = fig.add_subplot(gs[1])
ax2  = fig.add_subplot(gs[2])
ax3  = fig.add_subplot(gs[3])

###############################
# (A) SFFV-KRAB-dCas9-P2A-mCherry
# pHR-SFFV-KRAB-dCas9-P2A-mCherry (Addgene plasmid #60954)
###############################

filelist = \
["../../data/flow_cytometry/20191206_lenti-dCas9-mChr_CD4kd/HL60_lentidCas9-mCh_sgMYPT1_antiCD4_Data Source - 1.fcs",
 '../../data/flow_cytometry/20191218_lenti-dCas9-mCh-high_CD4kd/HL60_lentidCas9high-sgCD4_antiCD4.fcs',
 '../../data/flow_cytometry/20191218_lenti-dCas9-mCh-high_CD4kd/HL60_lentidCas9high-sgCD4_Ctrl.fcs']

sample = FCMeasurement(ID='ctrl', datafile=filelist[0])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] <= 6E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax0, color = colors[0],
            lw = 1, alpha = 1, legend = False)


sample = FCMeasurement(ID='ctrl', datafile=filelist[1])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] <= 6E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax0, color = colors[1],
            lw = 1, alpha = 1, legend = False )


sample = FCMeasurement(ID='ctrl', datafile=filelist[2])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.9E5]
sample = sample[sample['SSC-A'] <= 6E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax0, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)


###############################
# (B) dCas9-BFP-KRAB
# pHR-SFFV-dCas9-BFP-KRAB (Addgene plasmid #46911)
###############################
filelist = \
['../../data/flow_cytometry/20200309_CD4_knockdown/KW-dCas9-BFP-KRAB_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-dCas9-BFP-KRAB_gCD4_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-dCas9-BFP-KRAB_gCD4_antiCTRL.fcs']

sample = FCMeasurement(ID='ctrl', datafile=filelist[0])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax1, color = colors[0],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[1])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax1, color = colors[1],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[2])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax1, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)


###############################
# (Ci) Dox. inducible dCas9-KRAB-P2A-mCherry
# pAAVS1-NDi-CRISPRi (Gen1) (Addgene plasmid #73497)
###############################
filelist = \
['../../data/flow_cytometry/20200227_CD4kd/KW-NDi4x_0umDox_antiCD4.fcs',
'../../data/flow_cytometry/20200227_CD4kd/KW-NDi4x_0umDox_antiCTRL.fcs',
'../../data/flow_cytometry/20200227_CD4kd/KW-NDi4x_2umDox_antiCD4.fcs',
'../../data/flow_cytometry/20200227_CD4kd/KW-NDi4x_2umDox_antiCTRL.fcs']

sample = FCMeasurement(ID='ctrl', datafile=filelist[0])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.5E5]
sample = sample[sample['SSC-A'] <= 4E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax2, color = colors[0],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[2])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.5E5]
sample = sample[sample['SSC-A'] <= 4E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax2, color = colors[1],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[3])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 4E5]
sample = sample[sample['SSC-A'] >= 1.5E5]
sample = sample[sample['SSC-A'] <= 4E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax2, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)

###############################
# (Cii) clonal, Dox. inducible dCas9-KRAB-P2A-mCherry
# pAAVS1-NDi-CRISPRi (Gen1) (Addgene plasmid #73497)
###############################

filelist = \
['../../data/flow_cytometry/20200309_CD4_knockdown/KW-NDi-clone8_0umDox_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-NDi-clone8_gCD4_2umDox_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-NDi-clone8_gCD4_2umDox_antiCTRL.fcs',
]
sample = FCMeasurement(ID='ctrl', datafile=filelist[0])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax3, color = colors[0],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[1])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax3, color = colors[1],
            lw = 1, alpha = 1, legend = False,
           label = '0 Dox, anti-CD4')


sample = FCMeasurement(ID='ctrl', datafile=filelist[2])
sample = sample.data

# # gate out the noise and dead cells
sample = sample[sample['FSC-A'] >= 3.4E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample = sample[sample['SSC-A'] <= 5E5]
# log values
sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax3, color = 'grey',
            lw = 0, alpha = 0.3, legend = False, shade = True)


########################
########################

for ax_ in [ax0, ax1, ax2, ax3]:
    ax_.set_yticks([])
    ax_.grid(axis='x', color= 'grey', alpha  = 0.2)

    ax_.axes.get_yaxis().set_visible(False)
    ax_.set_xlabel(None)
    ax_.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                            2000,3000,4000,5000,6000,7000,8000,9000,
                            20000,30000,40000,50000,60000,70000,80000,90000,
                            200000,300000]), minor = True)
    ax_.set_xticks(np.log10([100,1000, 10000,100000]))
    ax_.set_xticklabels([])
    ax_.tick_params(length=8)
    ax_.set_xlim(2,4.3)

# for ax_ in [ax0, ax1, ax2]:
    ax_.tick_params(length=7, width=0.6)

plt.tight_layout()
fig.savefig('../../figures/FigS1_dCas9-KRAB_constructs.pdf')
