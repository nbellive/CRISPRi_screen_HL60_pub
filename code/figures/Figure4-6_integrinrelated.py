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

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
color = ['#738FC1', '#7AA974', '#CC462F', '#EAC264', '#97459B',
         '#7CD6C4', '#D87E6A', '#BCDB8A', '#BF78C4', '#9653C1']

df_2D = pd.read_csv('../../data/screen_summary/stats/sgRNA_avg/20220516_screen_log2fold_diffs_tracketch_all_sgRNA_pvalue.csv')
df_2D = df_2D[['gene','sgRNA','log2fold_diff_mean','exp','pvalue']]
df_2D = df_2D.drop_duplicates()

df_2D_err = pd.DataFrame()
files = glob.glob('../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_*_all.csv')
for f in files:
    df_2D_err = df_2D_err.append(pd.read_csv(f), ignore_index = True)
df_2D_err = df_2D_err[['sgRNA', 'gene', 'diff']]
df_2D_err.columns = ['sgRNA', 'gene', 'log2fold_diff_mean']
df_2D_ctrl = df_2D_err[df_2D_err.gene.str.contains('CONTROL')]


df_3D = pd.read_csv('../../data/screen_summary/stats/sgRNA_avg/20220516_screen_log2fold_diffs_3D_amoeboid_sgRNA_pvalue.csv')
df_3D = df_3D[['gene','sgRNA','log2fold_diff_mean','exp','pvalue']]
df_3D = df_3D.drop_duplicates()

df_3D_err = pd.read_csv('../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_all.csv')
df_3D_err =  df_3D_err.drop_duplicates()
df_3D_err = df_3D_err[['sgRNA', 'gene', 'diff', 'exp']]
df_3D_err.columns = ['sgRNA', 'gene', 'log2fold_diff_mean', 'exp']
df_3D_ctrl = df_3D_err[df_3D_err.gene.str.contains('CONTROL')]

df_3D = df_3D[df_3D.sgRNA.isin(df_3D_err.sgRNA.unique())]

#############################################
#  Figure on integrins and their trafficking
#############################################


fig = plt.figure(figsize=(8,8))
gs = GridSpec(nrows=8, ncols=3, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1], width_ratios=[0.2, 1, 3])

ax1 = fig.add_subplot(gs[:3,1])

ax2 = fig.add_subplot(gs[:2,2])
ax3 = fig.add_subplot(gs[2:4,2])

ax4 = fig.add_subplot(gs[4:6,:2])
ax5 = fig.add_subplot(gs[6:8,:2])

ax6 = fig.add_subplot(gs[4:6,2])
ax7 = fig.add_subplot(gs[6:8,2])

###############################
# A - 'top' hits - 2D
###############################
genes = [ 'ITGB2', 'ITGAM', '', 'APBB1IP', 'FERMT3', 'TLN1', 'RAP1A', 'RAP1B']
for i, val in enumerate(genes):
    if val == '':
        continue

    df_2D_temp = df_2D[df_2D.gene == val]
    df_2D_temp = df_2D_temp[np.abs(df_2D_temp.log2fold_diff_mean) == np.max(np.abs(df_2D_temp.log2fold_diff_mean.values))]

    sgRNA = df_2D_temp.sgRNA.unique()[0]
    df_err_ =  df_2D_err[df_2D_err.sgRNA == sgRNA].log2fold_diff_mean.std()/np.sqrt(len(df_2D_err[df_2D_err.sgRNA == sgRNA]))

    ax1.barh(len(genes) - i-0.5,
        df_2D_temp.log2fold_diff_mean.values[0],
        xerr = df_err_, color = '#7AA974')


ax1.set_yticks([])
ax1.set_xticks([-3,-2,-1,0])
ax1.set_xlim(-4,0.5)
ax1.set_ylim(0,8.25)
ax1.spines['left'].set_position(('data', 0))
ax1.set_xlabel(r'normalized'+'\n' + r'log$_2$ fold-change', fontsize = 10)

# Plot shaded region
###############################
x = np.arange(-1,10)
percentiles_arr = np.arange(5,95,0.5)
y_2D = np.percentile(df_2D_ctrl.groupby('sgRNA').log2fold_diff_mean.mean(), percentiles_arr)

# 2D data
ax1.fill_betweenx(x, np.min(y_2D), np.max(y_2D), alpha = 0.2, color = 'grey', zorder =0)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('top', size='7.5%', pad=0.05)

sns.distplot(y_2D, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 0, 'legend' : False},
                  ax = cax1, vertical=False, color = 'grey')
# cax1.set_ylim(-2,1)
cax1.set_xticks([-3,-2,-1,0])
cax1.set_xlim(-3.7,0.5)
cax1.set_xticks([])
cax1.set_yticks([])
cax1.axes.get_xaxis().set_visible(False)
cax1.axes.get_yaxis().set_visible(False)
cax1.spines['left'].set_visible(False)


###############################
# B - integrins all
###############################
genes = [ 'ITGA1', 'ITGA2', 'ITGA2B', 'ITGA3', 'ITGA4', 'ITGA5', 'ITGA6', 'ITGA7',
         'ITGA9', 'ITGA10', 'ITGA11', 'ITGAD', 'ITGAE', 'ITGAL', 'ITGAM', 'ITGAV', 'ITGAX',
         '',
         'ITGB1', 'ITGB2', 'ITGB3', 'ITGB4', 'ITGB7', 'ITGB8']
len_ITG = len(genes)

# Plot bar plot and error bars
###############################
for i, val in enumerate(genes):
    if val == '':
        continue

    df_2D_temp = df_2D[df_2D.gene == val]
    df_2D_temp = df_2D_temp[np.abs(df_2D_temp.log2fold_diff_mean) == np.max(np.abs(df_2D_temp.log2fold_diff_mean.values))]

    sgRNA = df_2D_temp.sgRNA.unique()[0]
    df_err_ =  df_2D_err[df_2D_err.sgRNA == sgRNA].log2fold_diff_mean.std()/np.sqrt(len(df_2D_err[df_2D_err.sgRNA == sgRNA]))

    ax2.bar(i,
        df_2D_temp.log2fold_diff_mean.values[0],
        yerr = df_err_, color = '#7AA974')

ax2.set_ylim(-3.7,1.0)
ax2.set_xlim(-1,len_ITG)
ax2.spines['bottom'].set_position(('data', 0))
ax2.set_xticks([])
ax2.set_ylabel(r'normalized'+'\n' + r'log$_2$ fold-change', fontsize = 10)

for i, val in enumerate(genes):
    if val == '':
        continue

    df_3D_temp = df_3D[df_3D.gene == val]
    df_3D_temp = df_3D_temp[np.abs(df_3D_temp.log2fold_diff_mean) == np.max(np.abs(df_3D_temp.log2fold_diff_mean.values))]

    sgRNA = df_3D_temp.sgRNA.unique()[0]
    df_err_ =  df_3D_err[df_3D_err.sgRNA == sgRNA].log2fold_diff_mean.std()/np.sqrt(len(df_3D_err[df_3D_err.sgRNA == sgRNA]))

    ax3.bar(i,
        df_3D_temp.log2fold_diff_mean.values[0],
        yerr = df_err_, color = '#EAC264')
    # print(val, ' : ', df_err_, np.sqrt(len(df_3D_err[df_3D_err.sgRNA == sgRNA])), df_3D_err[df_3D_err.sgRNA == sgRNA].log2fold_diff_mean.std() )
ax3.set_ylim(-2,1.0)
ax3.set_xlim(-1,len_ITG)
ax3.spines['bottom'].set_position(('data', 0))
ax3.set_xticks([])
ax3.set_ylabel(r'normalized'+'\n' + r'log$_2$ fold-change', fontsize = 10)

# Plot shaded region
###############################
x = np.arange(-1,len_ITG+2)
percentiles_arr = np.arange(5,95,0.5)
y_2D = np.percentile(df_2D_ctrl.groupby('sgRNA').log2fold_diff_mean.mean(), percentiles_arr)
y_3D = np.percentile(df_3D_ctrl.groupby('sgRNA').log2fold_diff_mean.mean(), percentiles_arr)

# 2D data
ax2.fill_between(x, np.min(y_2D), np.max(y_2D), alpha = 0.2, color = 'grey', zorder =0)
divider1 = make_axes_locatable(ax2)
cax2 = divider1.append_axes('right', size='7.5%', pad=0.05)

sns.distplot(y_2D, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 0, 'legend' : False},
                  ax = cax2, vertical=True, color = 'grey')
cax2.set_ylim(-2,1)
cax2.set_ylim(-3.7,1.0)
cax2.set_xticks([])
cax2.set_yticks([])
cax2.axes.get_xaxis().set_visible(False)
cax2.axes.get_yaxis().set_visible(False)
cax2.spines['bottom'].set_visible(False)

# 3D data
ax3.fill_between(x, np.min(y_3D), np.max(y_3D), alpha = 0.2, color = 'grey', zorder =0)
divider1 = make_axes_locatable(ax3)
cax3 = divider1.append_axes('right', size='7.5%', pad=0.05)

sns.distplot(y_3D, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 0, 'legend' : False},
                  ax = cax3, vertical=True, color = 'grey')
cax3.set_ylim(-2,1.0)
cax3.set_xticks([])
cax3.set_yticks([])
cax3.axes.get_xaxis().set_visible(False)
cax3.axes.get_yaxis().set_visible(False)
cax3.spines['bottom'].set_visible(False)
###################


###############################
# d - trafficking hits
###############################
# Retromer/Retreiver
# genes = ['VPS26A', 'VPS26B', 'VPS35','VPS29', 'COMMD1', 'CCDC22', 'CCDC92', 'C16orf62', 'DSCR3']
# HOPS/ CORVET
# genes = ['VPS39','VPS41', 'VPS18', 'VPS11', 'VPS16', 'VPS18',  'VPS33A', 'VPS33B','VPS8' ]

genes = ['VPS26A', 'VPS26B', 'VPS35','VPS29', 'COMMD1', 'CCDC22', 'CCDC92', 'C16orf62', 'DSCR3',
        '', '',
        'VPS39','VPS41', 'VPS18', 'VPS11', 'VPS16', 'VPS18',  'VPS33A', 'VPS33B','VPS8']

for i, val in enumerate(genes):
    if val == '':
        continue

    df_2D_temp = df_2D[df_2D.gene == val]
    df_2D_temp = df_2D_temp[np.abs(df_2D_temp.log2fold_diff_mean) == np.max(np.abs(df_2D_temp.log2fold_diff_mean.values))]

    sgRNA = df_2D_temp.sgRNA.unique()[0]
    df_err_ =  df_2D_err[df_2D_err.sgRNA == sgRNA].log2fold_diff_mean.std()/np.sqrt(len(df_2D_err[df_2D_err.sgRNA == sgRNA]))

    ax6.bar(i,
        df_2D_temp.log2fold_diff_mean.values[0],
        yerr = df_err_, color = '#7AA974')

ax6.set_ylim(-3,1)
ax6.set_xlim(-1,len_ITG)
ax6.spines['bottom'].set_position(('data', 0))
ax6.set_xticks([])
ax6.set_ylabel(r'normalized'+'\n' + r'log$_2$ fold-change', fontsize = 10)

for i, val in enumerate(genes):
    if val == '':
        continue
    df_3D_temp = df_3D[df_3D.gene == val]
    df_3D_temp = df_3D_temp[np.abs(df_3D_temp.log2fold_diff_mean) == np.max(np.abs(df_3D_temp.log2fold_diff_mean.values))]

    sgRNA = df_3D_temp.sgRNA.unique()[0]
    df_err_ =  df_3D_err[df_3D_err.sgRNA == sgRNA].log2fold_diff_mean.std()/np.sqrt(len(df_3D_err[df_3D_err.sgRNA == sgRNA]))

    ax7.bar(i,
        df_3D_temp.log2fold_diff_mean.values[0],
        yerr = df_err_, color = '#EAC264')

ax7.set_ylim(-3,1)
ax7.set_xlim(-1,len_ITG)
ax7.spines['bottom'].set_position(('data', 0))
ax7.set_xticks([])
ax7.set_ylabel(r'normalized'+'\n' + r'log$_2$ fold-change', fontsize = 10)


# Plot shaded region
###############################
x = np.arange(-1,len_ITG+2)
percentiles_arr = np.arange(5,95,0.5)
y_2D = np.percentile(df_2D_ctrl.groupby('sgRNA').log2fold_diff_mean.mean(), percentiles_arr)
y_3D = np.percentile(df_3D_ctrl.groupby('sgRNA').log2fold_diff_mean.mean(), percentiles_arr)

# Retromer/Retreiver - 2D
ax6.fill_between(x, np.min(y_2D), np.max(y_2D), alpha = 0.2, color = 'grey', zorder =0)
divider1 = make_axes_locatable(ax6)
cax3 = divider1.append_axes('right', size='7.5%', pad=0.05)

sns.distplot(y_2D, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 0, 'legend' : False},
                  ax = cax3, vertical=True, color = 'grey')
cax3.set_ylim(-3,1)
cax3.set_xticks([])
cax3.set_yticks([])
cax3.axes.get_xaxis().set_visible(False)
cax3.axes.get_yaxis().set_visible(False)
cax3.spines['bottom'].set_visible(False)

# Retromer/Retreiver - 3D
ax7.fill_between(x, np.min(y_3D), np.max(y_3D), alpha = 0.2, color = 'grey', zorder =0)
divider1 = make_axes_locatable(ax7)
cax4 = divider1.append_axes('right', size='7.5%', pad=0.05)

sns.distplot(y_3D, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 0, 'legend' : False},
                  ax = cax4, vertical=True, color = 'grey')
cax4.set_ylim(-3,1)
cax4.set_xticks([])
cax4.set_yticks([])
cax4.axes.get_xaxis().set_visible(False)
cax4.axes.get_yaxis().set_visible(False)
cax4.spines['bottom'].set_visible(False)
###################

###############################
# c - integrins migration data
###############################
files_bay_2d = glob.glob('../../data/processed_tracking_bayesian/20220920_2D_filtered_avg*all*')

df_Bayes_2d = pd.DataFrame()
for f in files_bay_2d:
    df_temp = pd.read_csv(f)
    df_Bayes_2d = df_Bayes_2d.append(df_temp, ignore_index = True)
df_Bayes_2d = df_Bayes_2d[df_Bayes_2d.celltype.isin(['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgITGB2'])]
df_Bayes_2d = df_Bayes_2d[['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'speed ($\mu$m/sec)', 'trial']]
df_Bayes_2d.columns = ['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'average_speed', 'trial']

files_bay_3d = glob.glob('../../data/processed_tracking_bayesian/20220920_3D_filtered_*avg*all*')
df_Bayes_3d = pd.DataFrame()
for f in files_bay_3d:
    df_temp = pd.read_csv(f)
    df_Bayes_3d = df_Bayes_3d.append(df_temp, ignore_index = True)
df_Bayes_3d = df_Bayes_3d[df_Bayes_3d.celltype.isin(['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgITGB2', 'HL-60KW_SC575_sgITGA1'])]
df_Bayes_3d = df_Bayes_3d[df_Bayes_3d.concentration == '0.75mgml']
df_Bayes_3d = df_Bayes_3d[['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'speed ($\mu$m/sec)', 'trial']]
df_Bayes_3d.columns = ['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'average_speed', 'trial']

########
# speed
########
lines_2D = ['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgITGB2' ]
vp_2d_s = np.empty_like(np.array([ax4,ax4]))
lines_3D = ['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgITGB2', 'HL-60KW_SC575_sgITGA1']
vp_3d_s = np.empty_like(np.array([ax4,ax4, ax4]))

pos = [0.5, 1, 1.75, 2.25, 2.75]
for i, ct in enumerate(lines_2D):
    y = df_Bayes_2d[df_Bayes_2d.celltype == ct]['average_speed']
    vp_2d_s[i] = ax4.violinplot(y, positions = [pos[i]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#D5DDEA'

    for pc in vp_2d_s[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax4.hlines(y.median(), pos[i]-0.1,pos[i]+0.1, zorder=10, lw = 1.5)

    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        ax4.errorbar(np.random.normal(pos[i], 0.05, 1), d['average_speed'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)


for i, ct in enumerate(lines_3D):
    y = df_Bayes_3d[df_Bayes_3d.celltype == ct]['average_speed']
    vp_3d_s[i] = ax4.violinplot(y, positions = [pos[i+2]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#D5DDEA'

    for pc in vp_3d_s[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax4.hlines(y.median(), pos[i+2]-0.1,pos[i+2]+0.1, zorder=10, lw = 1.5)

    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        d_top = d.iloc[int(len(d)/2):]
        ax4.errorbar(np.random.normal(pos[i+2], 0.05, 1), d_top['average_speed'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)
        d_bottom = d.iloc[:int(len(d)/2)]
        ax4.errorbar(np.random.normal(pos[i+2], 0.05, 1), d_bottom['average_speed'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)

# ax.set_xlim(0,2)
ax4.set_ylim(0.0,0.4)
ax4.set_ylabel('average speed\n' + r'[$\mu$m/s]', fontsize = 10)
ax4.spines['bottom'].set_position(('data', 0))
ax4.set_xticks([])

########
# persistence
########
vp_2d_p = np.empty_like(np.array([ax4,ax4]))
vp_3d_p = np.empty_like(np.array([ax4,ax4, ax4]))

pos = [0.5, 1, 1.75, 2.25, 2.75]
for i, ct in enumerate(lines_2D):
    y = df_Bayes_2d[df_Bayes_2d.celltype == ct]['average_persistence']
    vp_2d_p[i] = ax5.violinplot(y, positions = [pos[i]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#D5DDEA'

    for pc in vp_2d_p[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax5.hlines(y.median(), pos[i]-0.1,pos[i]+0.1, zorder=10, lw = 1.5)

    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        ax5.errorbar(np.random.normal(pos[i], 0.05, 1), d['average_persistence'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)

for i, ct in enumerate(lines_3D):
    y = df_Bayes_3d[df_Bayes_3d.celltype == ct]['average_persistence']
    vp_3d_p[i] = ax5.violinplot(y, positions = [pos[i+2]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#D5DDEA'

    for pc in vp_3d_p[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax5.hlines(y.median(), pos[i+2]-0.1,pos[i+2]+0.1, zorder=10, lw = 1.5)

    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        d_top = d.iloc[int(len(d)/2):]
        ax5.errorbar(np.random.normal(pos[i+2], 0.05, 1), d_top['average_persistence'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)
        d_bottom = d.iloc[:int(len(d)/2)]
        ax5.errorbar(np.random.normal(pos[i+2], 0.05, 1), d_bottom['average_persistence'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.3)

# ax.set_xlim(0,2)
ax5.set_ylim(-0.5,1.0)
ax5.set_ylabel('average persistence', fontsize = 10)
ax5.spines['bottom'].set_position(('data', 0))
ax5.set_xticks([])


########################
########################
# statistics
########################
########################
import scipy

param = 'average_speed'
exp_means_2d_wt = []
exp_means_2d_b2 = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_b2 = np.append(exp_means_2d_b2, d[param].mean())

print('2D, speed, ITGB2', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_b2))

exp_means_3d_wt = []
exp_means_3d_b2 = []
exp_means_3d_GA1 = []
for i, ct in enumerate(lines_3D):
    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        d_top = d.iloc[int(len(d)/2):]
        d_bottom = d.iloc[:int(len(d)/2)]
        if i == 0:
            exp_means_3d_wt = np.append(np.append(exp_means_3d_wt, d_top[param].mean()),d_bottom[param].mean())
        elif i == 1:
            exp_means_3d_b2 = np.append(np.append(exp_means_3d_b2, d_top[param].mean()),d_bottom[param].mean())
        elif i == 2:
            exp_means_3d_GA1 = np.append(np.append(exp_means_3d_GA1, d_top[param].mean()),d_bottom[param].mean())

print('3D, speed, ITGB2', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_b2))
print('3D, speed, ITGA1', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_GA1))
print('')
#perform one-way ANOVA
a = exp_means_3d_wt
b = exp_means_3d_b2
c = exp_means_3d_GA1
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'speed' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'ITGB2',
                                'speed' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'ITGA1',
                                'speed' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['speed'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)



param = 'average_persistence'
exp_means_2d_wt = []
exp_means_2d_b2 = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_b2 = np.append(exp_means_2d_b2, d[param].mean())

print('2D, persistence, ITGB2', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_b2))

exp_means_3d_wt = []
exp_means_3d_b2 = []
exp_means_3d_GA1 = []
for i, ct in enumerate(lines_3D):
    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        d_top = d.iloc[int(len(d)/2):]
        d_bottom = d.iloc[:int(len(d)/2)]
        if i == 0:
            exp_means_3d_wt = np.append(np.append(exp_means_3d_wt, d_top[param].mean()),d_bottom[param].mean())
        elif i == 1:
            exp_means_3d_b2 = np.append(np.append(exp_means_3d_b2, d_top[param].mean()),d_bottom[param].mean())
        elif i == 2:
            exp_means_3d_GA1 = np.append(np.append(exp_means_3d_GA1, d_top[param].mean()),d_bottom[param].mean())

print('3D, persistence, ITGB2', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_b2))
print('3D, persistence, ITGA1', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_GA1))



#perform one-way ANOVA
a = exp_means_3d_wt
b = exp_means_3d_b2
c = exp_means_3d_GA1
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'persistence' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'ITGB2',
                                'persistence' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'ITGA1',
                                'persistence' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['persistence'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)


plt.tight_layout()
fig.savefig('../../figures/Figure6_migration_integrins_.pdf', bbox_inches='tight')
