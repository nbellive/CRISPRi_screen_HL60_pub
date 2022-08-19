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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
color = ['#738FC1', '#7AA974', '#CC462F', '#EAC264', '#97459B',
         '#7CD6C4', '#D87E6A', '#BCDB8A', '#BF78C4', '#9653C1']

# colors2 = sns.color_palette("Set2")
#
# cell_lines_colors = {'sgCtrl1' : '#B8BABC',
#   'sgFLCN': colors2[6],
#   'sgLAMTOR1': colors2[0],
#   'sgTSC1' :  '#738FC1',
#   'sgRICTOR' :  '#738FC1'}

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
#  Figure 2: Summary of differentiattion screen results
# (A) Pathway enrichment
# (B) Overlap across screens
# (D,E) Look at cell density and survival in differentiation hits (density in D, survival in E)
#############################################


fig = plt.figure(figsize=(6,7.5))
gs = GridSpec(nrows=2, ncols=4, height_ratios=[1, 0.5], width_ratios=[0.1, 1, 0.1, 1])

ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,:2])
ax3 = fig.add_subplot(gs[1,2:])
# ax3 = fig.add_subplot(gs[2,1])

#############################
# B - bar plots based on pathway enrichment analysis (seperate plots for
# positive and negative)
###############################

# positive
pw_pos_dict = {'Mitochondrial translation' : 1E-7,
 'Respiratory electron transport chain' : 1E-7,
 'rRNA processing in nucleolus and cytosol' : 1E-7,
 'mTOR signaling and lysosome localization' : 1E-7,
 'Nucleosome, chromatin disassembly' : 6.685792E-6,
 'Protein targeting to peroxisome' : 0.003,
 'PI3K, AKT Signaling' : 0.008,
 'mRNA, catabolic processes' : 0.01,
 'Ubiquinone metabolic process' : 0.020,
 'Chromatin remodeling' : 0.029,
 'Regulation of hematopoietic progenitor cell differentiation' : 0.058}

# negative
pw_neg_dict = {'Transcription, mRNA processing' : 6.339028E-5,
 'protein acetylation' : 0.002,
 'autophagy of mitochondria' : 0.004,
 'autophagosome assembly' : 0.009,
 'Rho GTPases activate ROCKs' : 0.014,
 'Cortical cytoskeleton organization' : 0.016,
 'H3-K4 methylation' : 0.05,
 'Receptor signaling via JAK-STAT' : 0.051,
 'IL2-mediated signaling' : 0.054,
 'transcriptional regulation of granulopoiesis' : 0.054,
 'FGFR1 signaling' : 0.070}

for i, val in enumerate(pw_pos_dict):
    ax1.barh(len(pw_neg_dict)+len(pw_pos_dict) - i, -np.log10(pw_pos_dict[val]), color = '#1B4074')

for i, val in enumerate(pw_neg_dict):
    ax1.barh(i, np.log10(pw_neg_dict[val]), color = '#8E1E1E')

ax1.set_yticks([])
ax1.set_xlabel(r'signed log$_{10}$(adj.'
                '\np-value)')

ax1.spines['left'].set_position(('data', 0))
# ax4.spines['bottom'].set_position(('data', 0))

###############################
# D
###############################
# some  useful dictionaries

cell_lines_sg = ['AGGGCACCCGGTTCATACGC',
'GTAGGCGGAGAGGTCAATGG',
'CCCAGGGCTCCTGTAGCTCA',
'GCCCGGGTTCAGGCTCTCAG',
'GACTGTGAGGTAAACAGCTG',
'GCTGCTGTAGCAGCACCCCA',
'CGGGCTTACCTCGTACTCGG']

cell_lines_sg_dict = {'sgCtrl1': 'AGGGCACCCGGTTCATACGC',
 'sgCEBPE': 'GTAGGCGGAGAGGTCAATGG',
 'sgSPI1': 'CCCAGGGCTCCTGTAGCTCA',
 'sgFLCN': 'GCCCGGGTTCAGGCTCTCAG',
 'sgTSC1': 'GACTGTGAGGTAAACAGCTG',
 'sgLAMTOR1': 'GCTGCTGTAGCAGCACCCCA',
 'sgRICTOR': 'CGGGCTTACCTCGTACTCGG'}


markers = ["v", "^", "<", ">", "s", "p", "P", "*","+", "x", "D", "H"]

cell_lines_marker_dict = {'sgCtrl1': 'o',
 'sgCEBPE' :  "v",
 'sgSPI1': "<",
 'sgFLCN': ">",
 'sgTSC1': "s",
 'sgLAMTOR1': "p",
 'sgRICTOR': "P"}


# load in the data
df_screen = pd.read_csv('../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_differentiation_sgRNA_means.csv')


# we only care about  the sgRNA  that we've made cell lines for
print(df_screen.head())
df_screen = df_screen[df_screen.sgRNA.isin(cell_lines_sg)]
df_screen = df_screen[['gene','sgRNA','log2fold_diff_mean']]


df = pd.read_csv('../../data/screen_followups/20211122_proliferation_curves.csv')
df_exps = df[df.date.isin(['20111116', '20111121'])]
df_exps = df_exps[df_exps.DMSO == True]

# generate plots
for gene, d in df_exps[df_exps.date == df_exps.date.max()].groupby('cell_line'):
    if gene == 'sgCtrl1':
        ax2.errorbar(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean.mean(),
                    d.effective_density_1E6_ml.mean(),
                     yerr = d.effective_density_1E6_ml.std(),
                    xerr = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean.std(),#/np.sqrt(len(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean)),
                     markersize = 9, marker = cell_lines_marker_dict[gene], color = '#B8BABC',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1, label = gene)
        ctrl_value = d.effective_density_1E6_ml.mean()

    else:
        ax2.errorbar(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean.mean(),
                    d.effective_density_1E6_ml.mean(),
                    yerr = d.effective_density_1E6_ml.std(),
                    xerr = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean.std(),#/np.sqrt(len(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff_mean)),
                    marker =  cell_lines_marker_dict[gene],
                    color  = cell_lines_colors[gene], markersize = 9, markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1, label = gene)



ax2.set_xlabel(r'log$_2$ fold-change'
                '\n(differentiation screen data)')
ax2.set_ylabel('dHL-60 cell density\n'
              r'[$x10^{6}$ cells/ml]')

ax2.set_xlim(-1.5,1.5)
ax2.set_ylim(0.5,1.3)
ax2.set_xticks([-1,0,1])
ax2.axhline(ctrl_value,xmin =-3.25, xmax = 1.5, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )
ax2.axvline(0,ymin = 0, ymax = 1, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )

# ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')#, fontsize = 14)

#############################
# (E) Cell lifetime experiments
#############################
#############################

files = glob.glob('../../data/screen_followups/20211122_proliferation_curves.csv')
df_growth = pd.read_csv(files[0])
df_growth.head()

df_growth = df_growth[df_growth.cell_line != 'sgSPI1']
df_growth = df_growth[df_growth.cell_line != 'sgCEBPE']


df_growth.date.unique()
day_dict = {'20111116' : 0,
            '20111117': 1,
            '20111118': 2,
            '20111119': 3,
            '20111120': 4,
            '20111121': 5,
            '20111123': 7,
            '20111126': 10,
            '20211129': 13}

df_growth['trial'] = 1
df_growth['date_'] = df_growth['date'].astype('str')

df_growth['day_diff'] = df_growth['date_'].map(day_dict)


cell_lines_marker_dict = {'sgCtrl1': 'o',
 'sgCEBPE' :  "v",
 'sgSPI1': "<",
 'sgFLCN': ">",
 'sgTSC1': "s",
 'sgLAMTOR1': "p",
 'sgRICTOR': "P"}


date_dict = dict(zip(df_growth[df_growth.DMSO == True].date.unique(), 24*np.array([0,1,2,3,4,5,7,10,13]) + [13,10,11.5,13,14,17,11,9,9]))
day_dict = dict(zip(df_growth.date.unique(), np.arange(7)))
cell_line_dict = dict(zip(df_growth.cell_line.unique(), colors))

for g,d in df_growth[df_growth.DMSO == True][df_growth.day_diff>=4].groupby(['date', 'cell_line']):
    t = d.day_diff.unique()
    if 'Ctrl' in g[1]:
        color = '#B8BABC'
    else:
        color = '#738FC1'


    d_temp = df_growth[df_growth.DMSO == True]
    d_temp = d_temp[d_temp.cell_line == g[1]]
    d_temp = d_temp[d_temp.date == 20111120]
    yerr = np.abs((d.effective_density_1E6_ml.mean()/d_temp.effective_density_1E6_ml.mean()))*np.sqrt((d['effective_density_1E6_ml'].std()/d['effective_density_1E6_ml'].mean())**2 + (d_temp['effective_density_1E6_ml'].std()/d_temp['effective_density_1E6_ml'].mean())**2)
    ax3.errorbar(t, d.effective_density_1E6_ml.mean()/d_temp.effective_density_1E6_ml.mean(),
                yerr = yerr, label = g[1],
                markersize = 9, marker = cell_lines_marker_dict[g[1]], color  = cell_lines_colors[g[1]],
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1)


for g,d in df_growth[df_growth.DMSO == True][df_growth.day_diff>=4].groupby(['cell_line']):
    if g != 'sgCtrl1':
        continue
    lw = 3
    d_temp = df_growth[df_growth.DMSO == True]
    d_temp = d_temp[d_temp.cell_line == g]
    d_temp = d_temp[d_temp.date == 20111120]

    t = d.groupby('day_diff').day_diff.unique().values
    ax3.plot(t, d.groupby('day_diff').effective_density_1E6_ml.mean().values/d_temp.effective_density_1E6_ml.mean(),
               color = cell_lines_colors[g], label = None, lw = lw)


for g,d in df_growth[df_growth.DMSO == True][df_growth.day_diff>=4].groupby(['cell_line']):
    if g == 'sgCtrl1':
        continue
    else:
        lw = 1

    d_temp = df_growth[df_growth.DMSO == True]
    d_temp = d_temp[d_temp.cell_line == g]
    d_temp = d_temp[d_temp.date == 20111120]

    t = d.groupby('day_diff').day_diff.unique().values
    ax3.plot(t, d.groupby('date').effective_density_1E6_ml.mean().values/d_temp.effective_density_1E6_ml.mean(),
               color  = cell_lines_colors[g], label = None, lw = lw)

# legend_without_duplicate_labels(ax3)
ax3.set_xlabel('time post-differentiation [day] ')
ax3.set_ylabel('dHL-60 relative cell density')
ax3.set_xlim(3.5,14)
ax3.set_ylim(0,1.15)

for ax_ in [ax1, ax2, ax3]:
    for axis in ['bottom','left']:
        ax_.spines[axis].set_linewidth(0.6)
    ax_.tick_params(width=0.6)


plt.tight_layout()
fig.savefig('../../figures/Fig2_differentiation.pdf', bbox_inches='tight')
