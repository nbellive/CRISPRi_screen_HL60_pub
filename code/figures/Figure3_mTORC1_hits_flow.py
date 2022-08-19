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

#############################################
#  Figure 3: Characterization of cells following KD of LAMTOR1 and FLCN
# (A, B, C) Characterization of differentiation markers
# (D) phase images of sgFLCN, sgLAMTOR1, and sgControl1
# (E) migration analysis (different .py file)
# (F) chemotaxis of FLCN (different .py file)
#############################################

# fig = plt.figure(figsize=(5,6))
fig = plt.figure(figsize=(5,4))
gs = GridSpec(nrows=2, ncols=3, height_ratios=[1.25, 2], width_ratios=[1, 1, 1])


ax0  = fig.add_subplot(gs[0,0])
ax1  = fig.add_subplot(gs[0,1])
ax2  = fig.add_subplot(gs[1,0:2])
ax3  = fig.add_subplot(gs[1,2])


###############################
# top row, undifferentiated and differentiated cells
###############################

files = glob.glob('../../data/flow_cytometry/20201008_surface_markers/*CD11b*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[0])
sample = sample[sample['FSC-A'] >= 2.5E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample['FITC-A_l'] = np.log10(sample['FITC-A'][sample['FITC-A'] > 0])
sample['APC-A_l'] = np.log10(sample['APC-A'][sample['APC-A']>0])

sns.kdeplot(sample["FITC-A_l"], ax = ax0, color = 'k', lw = 1,
            ls = '--', alpha = 1, legend = False, shade = False)

sample = FCMeasurement(ID='ctrl', datafile=files[1])
sample = sample[sample['FSC-A'] >= 2.5E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample['FITC-A_l'] = np.log10(sample['FITC-A'][sample['FITC-A'] > 0])
sample['APC-A_l'] = np.log10(sample['APC-A'][sample['APC-A']>0])

g1 = sns.kdeplot(sample["FITC-A_l"], ax = ax0, color = 'k', lw = 1,
            ls = '-', alpha = 1, legend = False)

g1.set(xlabel=None, ylabel=None)


ax0.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                        2000,3000,4000,5000,6000,7000,8000,9000,
                        20000,30000,40000,50000,60000,70000,80000,90000,
                        200000,300000]), minor = True)
ax0.set_xticks(np.log10([100,1000, 10000,100000]))
ax0.set_xlim(2,5.5)
ax0.set_xticklabels([])
ax0.set_yticks([])
ax0.set_yticklabels([])


#########################

files = glob.glob('../../data/flow_cytometry/20201008_surface_markers/*fMLP*.fcs')

sample = FCMeasurement(ID='ctrl', datafile=files[0])
sample = sample[sample['FSC-A'] >= 2.5E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample['FITC-A_l'] = np.log10(sample['FITC-A'][sample['FITC-A'] > 0])
sample['APC-A_l'] = np.log10(sample['APC-A'][sample['APC-A']>0])

# sns.kdeplot(sample["APC-A_l"], ax = ax1, color = 'grey',
#             lw = 0, alpha = 0.3, legend = False, shade = True)
sns.kdeplot(sample["APC-A_l"], ax = ax1, color = 'k', lw = 1,
            ls = '--', alpha = 1, legend = False, shade = False)

sample = FCMeasurement(ID='ctrl', datafile=files[1])
sample = sample[sample['FSC-A'] >= 2.5E5]
sample = sample[sample['SSC-A'] >= 2E5]
sample['FITC-A_l'] = np.log10(sample['FITC-A'][sample['FITC-A'] > 0])
sample['APC-A_l'] = np.log10(sample['APC-A'][sample['APC-A']>0])

# g2 = sns.kdeplot(sample["APC-A_l"], ax = ax1, color = 'grey',
#             lw = 0, alpha = 0.3, legend = False, shade = True)
g2 = sns.kdeplot(sample["APC-A_l"], ax = ax1, color = 'k', lw = 1,
            ls = '-', alpha = 1, legend = False, shade = False)

g2.set(xlabel=None, ylabel=None)


ax1.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                        2000,3000,4000,5000,6000,7000,8000,9000,
                        20000,30000,40000,50000,60000,70000,80000,90000]), minor = True)
ax1.set_xticks(np.log10([100,1000, 10000,100000]))
ax1.set_xlim(2,4.6)
ax1.set_xticklabels([])
ax1.set_yticks([])
ax1.set_yticklabels([])


###############################
# bottom row, undifferentiated and differentiated cells
###############################

files_ = ['../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_sgCtrl1_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_LAMTOR1_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_FLCN_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_sgSPI1_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_RICTOR_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_TSC1_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_sgCEBPE_Data Source - 1.fcs',
     '../../data/flow_cytometry/20211121_diffScreen_KD_lines/diff_sgCtrl1_isotypectrl_Data Source - 1.fcs']

sample1 = FCMeasurement(ID='ctrl', datafile=files_[0])
sample1 = sample1.data
# gate out the noise and dead cells
sample1 = sample1[sample1['FSC-A'] >= 1.8E5]
sample1 = sample1[sample1['FSC-A'] <= 4.1E5]
sample1 = sample1[sample1['SSC-A'] >= 0.3E5]
sample1 = sample1[sample1['SSC-A'] <= 1.5E5]
sample1['BB515-A_l'] = np.log10(sample1['BB515-A'][sample1['BB515-A'] > 0])
sample1['AF 647-A_l'] = np.log10(sample1['AF 647-A'][sample1['AF 647-A']>0])
sample1['cell_line'] = 'Control'
sample_wt = sample1

sample2 = FCMeasurement(ID='ctrl', datafile=files_[1])
sample2 = sample2.data
# gate out the noise and dead cells
sample2 = sample2[sample2['FSC-A'] >= 1.8E5]
sample2 = sample2[sample2['FSC-A'] <= 4.1E5]
sample2 = sample2[sample2['SSC-A'] >= 0.3E5]
sample2 = sample2[sample2['SSC-A'] <= 1.5E5]
sample2['BB515-A_l'] = np.log10(sample2['BB515-A'][sample2['BB515-A'] > 0])
sample2['AF 647-A_l'] = np.log10(sample2['AF 647-A'][sample2['AF 647-A']>0])
sample2['cell_line'] = 'LAMTOR1'

sample3 = FCMeasurement(ID='ctrl', datafile=files_[2])
sample3 = sample3.data
# gate out the noise and dead cells
sample3 = sample3[sample3['FSC-A'] >= 1.8E5]
sample3 = sample3[sample3['FSC-A'] <= 4.1E5]
sample3 = sample3[sample3['SSC-A'] >= 0.3E5]
sample3 = sample3[sample3['SSC-A'] <= 1.5E5]
sample3['BB515-A_l'] = np.log10(sample3['BB515-A'][sample3['BB515-A'] > 0])
sample3['AF 647-A_l'] = np.log10(sample3['AF 647-A'][sample3['AF 647-A']>0])
sample3['cell_line'] = 'FLCN'

sample4 = FCMeasurement(ID='ctrl', datafile=files_[3])
sample4 = sample4.data
# gate out the noise and dead cells
sample4 = sample4[sample4['FSC-A'] >= 1.8E5]
sample4 = sample4[sample4['FSC-A'] <= 4.1E5]
sample4 = sample4[sample4['SSC-A'] >= 0.3E5]
sample4 = sample4[sample4['SSC-A'] <= 1.5E5]
sample4['BB515-A_l'] = np.log10(sample4['BB515-A'][sample4['BB515-A'] > 0])
sample4['AF 647-A_l'] = np.log10(sample4['AF 647-A'][sample4['AF 647-A']>0])
sample4['cell_line'] = 'SPI1'

sample = sample1.append(sample2, ignore_index = True)
sample = sample.append(sample3, ignore_index = True)
sample = sample.append(sample4, ignore_index = True)

sample4 = FCMeasurement(ID='ctrl', datafile=files_[4])
sample4 = sample4.data
# gate out the noise and dead cells
sample4 = sample4[sample4['FSC-A'] >= 1.8E5]
sample4 = sample4[sample4['FSC-A'] <= 4.1E5]
sample4 = sample4[sample4['SSC-A'] >= 0.3E5]
sample4 = sample4[sample4['SSC-A'] <= 1.5E5]
sample4['BB515-A_l'] = np.log10(sample4['BB515-A'][sample4['BB515-A'] > 0])
sample4['AF 647-A_l'] = np.log10(sample4['AF 647-A'][sample4['AF 647-A']>0])
sample4['cell_line'] = 'RICTOR'
sample = sample.append(sample4, ignore_index = True)

sample4 = FCMeasurement(ID='ctrl', datafile=files_[5])
sample4 = sample4.data
# gate out the noise and dead cells
sample4 = sample4[sample4['FSC-A'] >= 1.8E5]
sample4 = sample4[sample4['FSC-A'] <= 4.1E5]
sample4 = sample4[sample4['SSC-A'] >= 0.3E5]
sample4 = sample4[sample4['SSC-A'] <= 1.5E5]
sample4['BB515-A_l'] = np.log10(sample4['BB515-A'][sample4['BB515-A'] > 0])
sample4['AF 647-A_l'] = np.log10(sample4['AF 647-A'][sample4['AF 647-A']>0])
sample4['cell_line'] = 'TSC1'
sample = sample.append(sample4, ignore_index = True)

sample4 = FCMeasurement(ID='ctrl', datafile=files_[6])
sample4 = sample4.data
# gate out the noise and dead cells
sample4 = sample4[sample4['FSC-A'] >= 1.8E5]
sample4 = sample4[sample4['FSC-A'] <= 4.1E5]
sample4 = sample4[sample4['SSC-A'] >= 0.3E5]
sample4 = sample4[sample4['SSC-A'] <= 1.5E5]
sample4['BB515-A_l'] = np.log10(sample4['BB515-A'][sample4['BB515-A'] > 0])
sample4['AF 647-A_l'] = np.log10(sample4['AF 647-A'][sample4['AF 647-A']>0])
sample4['cell_line'] = 'CEBPE'
sample = sample.append(sample4, ignore_index = True)

sample = sample.dropna()
sample_wt = sample_wt.dropna()

features = ['BB515-A_l', 'AF 647-A_l']
# Separating out the features
x = sample_wt.loc[:, features].values
# Separating out the target
y = sample_wt.loc[:,['cell_line']].values

# Perform PCA on only the wild-type (sgControl) data
pca = PCA(n_components=2)
principalComponents_wt = pca.fit_transform(x)

x = sample.loc[:, features].values
# Separating out the target
y = sample.loc[:,['cell_line']].values

principalComponents = pca.transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, sample[['cell_line']]], axis = 1)

targets = ['LAMTOR1', 'FLCN', 'Control', 'TSC1', 'CEBPE', 'RICTOR', 'SPI1']

count = 0
for target in targets:

    indicesToKeep = finalDf['cell_line'] == target

    # Number of replicatess
    n_reps = 2000

    # Initialize bootstrap replicas array
    bs_reps = np.empty(n_reps)

    # Compute replicates
    bs_reps = np.array(
        [draw_bs_rep(finalDf.loc[indicesToKeep, 'principal component 1'][:500], np.mean, rg) for _ in range(n_reps)]
    )

    # Compute the confidence interval
    conf_int = np.percentile(bs_reps, [0.5, 99.5])

    mean = finalDf.loc[indicesToKeep, 'principal component 1'][:500].mean()

    ax3.hlines(count, xmin = -1*conf_int[1], xmax = -1*conf_int[0],
                linewidth=1, color = 'k', label = None)
    count += 1


    y =  -1*finalDf.loc[indicesToKeep, 'principal component 1'][:1000]
    vp_ = ax3.violinplot(y, positions = [count-1], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2,  vert=False)

    for pc in vp_['bodies']:
        pc.set_color(cell_lines_colors['sg'+target])
        pc.set_alpha(0.5)

ax3.set_yticks(np.arange(len(targets)))
ax3.set_yticklabels(targets)
ax3.set_xlim(-2.4, 1.5)

for x in [-2, 0]:
    ax3.vlines(x, -1,len(targets), alpha = 0.1, zorder=0)
ax3.set_ylim(-0.5, len(targets)-0.5)

#######################
# wild-type control 2d plot
#######################
features = ['BB515-A_l', 'AF 647-A_l']
x = sample_wt.loc[:, features].values
y = sample_wt.loc[:,['cell_line']].values
pca_ = PCA(n_components=2)
principalComponents_ = pca_.fit_transform(x)
principalDf_ = pd.DataFrame(data = principalComponents,
            columns = ['principal component 1', 'principal component 2'])

sns.kdeplot(data=sample_wt, x="BB515-A_l", y="AF 647-A_l",
                 shade=True, thresh=.2, cmap="rocket_r", ax = ax2)


count = 0
for length, vector in zip(pca_.explained_variance_, pca_.components_):
    if count == 0:
        v = vector  * np.sqrt(length)
        x = (pca_.mean_[:] + 2*v)[0], (pca_.mean_[:] + v)[0], pca_.mean_[:][0], (pca_.mean_  -v)[0], (pca_.mean_  -2*v)[0]
        y = (pca_.mean_[:] + 2*v)[1], (pca_.mean_[:] + v)[1], pca_.mean_[:][1], (pca_.mean_  -v)[1], (pca_.mean_  -2*v)[1]
        ax2.plot(x,y, linewidth=1.5, color = 'gray', linestyle = '--', marker = "|")

    else:
        v = vector * np.sqrt(length)
        x = (pca_.mean_[:])[0], (pca_.mean_  +v)[0]
        y = (pca_.mean_[:])[1], (pca_.mean_  +v)[1]
        ax2.plot(x,y, linewidth=1.5, color = 'gray', linestyle = '--' )

    count += 1

ax2.set_aspect('equal')

#######################
# isotype 2d plot
#######################
sample_iso = FCMeasurement(ID='ctrl', datafile=files_[7])
sample_iso = sample_iso.data

# gate out the noise and dead cells
sample_iso = sample_iso[sample_iso['FSC-A'] >= 1.8E5]
sample_iso = sample_iso[sample_iso['FSC-A'] <= 4.1E5]
sample_iso = sample_iso[sample_iso['SSC-A'] >= 0.3E5]
sample_iso = sample_iso[sample_iso['SSC-A'] <= 1.5E5]
sample_iso['BB515-A_l'] = np.log10(sample_iso['BB515-A'][sample_iso['BB515-A'] > 0])
sample_iso['AF 647-A_l'] = np.log10(sample_iso['AF 647-A'][sample_iso['AF 647-A']>0])
sample_iso['cell_line'] = 'isotype'

g1 = sns.kdeplot(data=sample_iso, x="BB515-A_l", y="AF 647-A_l",
                 shade=True, thresh=.2, cmap="gray_r", ax =ax2, alpha = 0.6)
g1.set(xlabel=None, ylabel=None)

ax2.set_xticks(np.log10([60,70,80,100,200,300,400,500,600,700,800,900,
                        2000,3000,4000,5000,6000,7000,8000,9000,
                        20000,30000,40000,50000,60000,70000,80000,90000,
                        200000,300000]), minor = True)
ax2.set_xticks(np.log10([100,1000, 10000,100000]))
ax2.set_yticks(np.log10([30,40,50,60,70,80,90,200,300,400,500,600,700,800,900,
                        2000,3000, 4000,5000,6000,7000,8000,9000]), minor = True)
ax2.set_yticks(np.log10([100,1000, 10000]))
for y in np.log10([10,100,1000, 10000]):
    ax2.hlines(y, 2,5.5,linewidth=0.6, alpha = 0.1, zorder=20)
for x in np.log10([100, 1000, 10000,100000]):
    ax2.vlines(x, 1,4.5,linewidth=0.6, alpha = 0.1, zorder=20)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xlim(2,5)
ax2.set_ylim(2,3.5)

# for ax_ in [ax0, ax1, ax2, ax3]:
#     ax_.tick_params(width=0.6)
for ax_ in [ax0, ax1, ax2]:
    ax_.tick_params(length=7, width=0.6)

plt.tight_layout()
fig.savefig('../../figures/Fig3ABC_mTORCcandidates_flowonly.pdf')
