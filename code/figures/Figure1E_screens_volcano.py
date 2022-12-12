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

def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


# plt.style.use('styleNB')
plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']


#############################################
#  Figure 1: Summary of screen and overview of data
# (A-D) Experimental schematic;  plot to show knockdown in HL-60 cells; bulk migration results - fraction of cells migrated
# (E) Volcano plots to show the performance of the screens - showing the results/ statistical
# significance  of each of the proliferation, differentiation, and combined cell migration data
# (F) Upset plot to show overlap of identified genes across the screens (in a different .py file)
#############################################

fig = plt.figure(figsize=(9*0.75,3*0.75))

gs = GridSpec(nrows=1, ncols=6 , height_ratios=[1], width_ratios=[1, 0.25, 1, 0.25, 1, 0.25])

ax1 = fig.add_subplot(gs[0,0])
ax1_marg = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax2_marg = fig.add_subplot(gs[0,3])
ax3 = fig.add_subplot(gs[0,4])
ax3_marg = fig.add_subplot(gs[0,5])


###############################
# A-C - summary of screens - volcano plots
###############################

# load in the data
# df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_growth_gene_pvalues.csv')
df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_growth_means_pvalue.csv')

#  proliferation; # calculate fdr
df_ = df[df.exp == 'growth']
df_['fdr'] = fdr(df_.pvalue)

xmin = np.min(df_.log2fold_diff_mean)
xmax = np.max(df_.log2fold_diff_mean)
ax1.hlines(-np.log10(0.05), xmin = -4, xmax = 4, color = 'k',
           linestyles='--', alpha = 0.5, linewidth = 1, zorder = 0)

df_exp = df_[~df_.gene.str.contains('CONTROL')]
df_ctrl = df_[df_.gene.str.contains('CONTROL')]

ax1.plot(df_exp.log2fold_diff_mean,
        -np.log10(df_exp.fdr), zorder = 10,
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax1.plot(df_ctrl.log2fold_diff_mean,
        -np.log10(df_ctrl.fdr), zorder = 10,
           color = 'grey',
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax1.text(-2, 2.5, len(df_exp[df_exp.log2fold_diff_mean<=0][df_exp.fdr<=0.05]))
ax1.text(2, 2.5, len(df_exp[df_exp.log2fold_diff_mean>=0][df_exp.fdr<=0.05]))

ax1.set_xlim(-2.5,2.5)
ax1.set_ylim(0,5.2)

sns.distplot(-np.log10(df_exp.fdr), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax1_marg, vertical=True)

#########################
#  differentiation; # plot
#########################
# df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_differentiation_gene_pvalues.csv')
df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')

#  differentiation; # calculate fdr
df_ = df[df.exp == 'differentiation']
df_['fdr'] = fdr(df_.pvalue)

# ax2.set_xlabel('normalized log2 fold-change', fontsize = 14)
# ax2.set_ylabel('-log10(adjusted P-value)', fontsize = 14)
xmin = np.min(df_.log2fold_diff_mean)
xmax = np.max(df_.log2fold_diff_mean)
ax2.hlines(-np.log10(0.05), xmin = -4, xmax = 4, color = 'k',
           linestyles='--', alpha = 0.5, linewidth = 1, zorder = 0)

df_exp = df_[~df_.gene.str.contains('CONTROL')]
df_ctrl = df_[df_.gene.str.contains('CONTROL')]

ax2.plot(df_exp.log2fold_diff_mean,
        -np.log10(df_exp.fdr), zorder = 10,
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax2.plot(df_ctrl.log2fold_diff_mean,
        -np.log10(df_ctrl.fdr), zorder = 10,
           color = 'grey',
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax2.text(-2, 2.5, len(df_exp[df_exp.log2fold_diff_mean<=0][df_exp.fdr<=0.05]))
ax2.text(2, 2.5, len(df_exp[df_exp.log2fold_diff_mean>=0][df_exp.fdr<=0.05]))
ax2.set_xlim(-2.5,2.5)
ax2.set_ylim(0,5.2)

sns.distplot(-np.log10(df_exp.fdr), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax2_marg, vertical=True)

#########################
#  migration; # plot
#########################
df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_migration_all_means_pvalue.csv')

df_ = df
df_['fdr'] = fdr(df_.pvalue)

# ax3.set_xlabel('normalized log2 fold-change', fontsize = 14)
# ax3.set_ylabel('-log10(adjusted P-value)', fontsize = 14)
xmin = np.min(df_.log2fold_diff_mean)
xmax = np.max(df_.log2fold_diff_mean)
ax3.hlines(-np.log10(0.05), xmin = -4, xmax = 4, color = 'k',
           linestyles='--', alpha = 0.5, linewidth = 1, zorder = 0)

df_exp = df_[~df_.gene.str.contains('CONTROL')]
df_ctrl = df_[df_.gene.str.contains('CONTROL')]

ax3.plot(df_exp.log2fold_diff_mean,
        -np.log10(df_exp.fdr), zorder = 10,
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax3.plot(df_ctrl.log2fold_diff_mean,
        -np.log10(df_ctrl.fdr), zorder = 10,
           color = 'grey',
           marker = 'o', lw = 0,markeredgecolor = 'k',
            markeredgewidth = 0.15, alpha = 0.5,
            markersize = 5)

ax3.text(-2, 2.5, len(df_exp[df_exp.log2fold_diff_mean<=0][df_exp.fdr<=0.05]))
ax3.text(2, 2.5, len(df_exp[df_exp.log2fold_diff_mean>=0][df_exp.fdr<=0.05]))

ax3.set_xlim(-2.5,2.5)
ax3.set_ylim(0,5.2)


sns.distplot(-np.log10(df_exp.fdr), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax3_marg, vertical=True)

for  ax_ in [ax1_marg, ax2_marg, ax3_marg]:
    ax_.get_xaxis().set_visible(False)
    ax_.get_yaxis().set_visible(False)
    ax_.set_xticklabels([])
    ax_.set_yticklabels([])

for  ax_ in [ax1, ax2, ax3, ax1_marg, ax2_marg, ax3_marg]:
    for axis in ['bottom','left']:
        ax_.spines[axis].set_linewidth(0.6)
    ax_.tick_params(width=0.6)

plt.tight_layout()
fig.savefig('../../figures/Fig1E_screens_volcano_.pdf')#, bbox_inches='tight')
