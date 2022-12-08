import numpy as np
import pandas as pd
import random
import glob
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

plt.style.use('styleNB.mplstyle')

import glob

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
#############################################
#  Figure S2: Comparison of proliferation screen results with Sanson et al.
#############################################

#############################################
# Load in Sanson et al. 2018 data
#############################################
files = glob.glob('../../../data/sanson_2018/sanson_2018_setA_rawreads.csv')
df_Sanson_raw = pd.read_csv(glob.glob('../../data/sanson_2018/sanson_2018_setA_rawreads.csv')[0])
df_Sanson_ann = pd.read_csv(glob.glob('../../data/sanson_2018/sanson_2018_setA_sgRNA_annotations.csv')[0])

# Make merged df with Sanson data and annotations
df_Sanson = pd.merge(df_Sanson_ann, df_Sanson_raw, on = 'sgRNA')

# Calculate normalized log2 fold-changes (relative to plasmid)
df_Sanson['HT29_RepA_diff'] = np.log2(df_Sanson['HT29_RepA'] + 32) - np.log2(df_Sanson['HT29_RepA'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))
df_Sanson['HT29_RepB_diff'] = np.log2(df_Sanson['HT29_RepB'] + 32) - np.log2(df_Sanson['HT29_RepB'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))
df_Sanson['HT29_RepC_diff'] = np.log2(df_Sanson['HT29_RepC'] + 32) - np.log2(df_Sanson['HT29_RepC'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))

df_Sanson['A375_RepA_diff'] = np.log2(df_Sanson['A375_RepA'] + 32) - np.log2(df_Sanson['A375_RepA'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))
df_Sanson['A375_RepB_diff'] = np.log2(df_Sanson['A375_RepB'] + 32) - np.log2(df_Sanson['A375_RepB'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))
df_Sanson['A375_RepC_diff'] = np.log2(df_Sanson['A375_RepC'] + 32) - np.log2(df_Sanson['A375_RepC'].median()) - (np.log2(df_Sanson['pDNA'] + 32) - np.log2(df_Sanson['pDNA'].median()))

# mean across replicates
df_Sanson['HT29_diff_mean'] = df_Sanson[['HT29_RepA_diff','HT29_RepB_diff','HT29_RepC_diff']].mean(axis=1)
df_Sanson['A375_diff_mean'] = df_Sanson[['A375_RepA_diff','A375_RepB_diff','A375_RepC_diff']].mean(axis=1)

# mean across sgRNA (3 per gene, Set A)
df_Sanson_avg = pd.DataFrame()
for g, d in df_Sanson.groupby('gene'):
    data_list = {'gene' : g,
                'HT29_diff_mean': d.HT29_diff_mean.mean(),
                'A375_diff_mean': d.A375_diff_mean.mean()}
    df_Sanson_avg = df_Sanson_avg.append(data_list, ignore_index = True)

# Control sgRNAs
df_Sanson_controls = pd.DataFrame()
for sg, d in df_Sanson[df_Sanson.gene == 'CONTROL'].groupby('sgRNA'):
    data_list = {'sgRNA' : sg,
                'HT29_diff_mean': d.HT29_diff_mean.mean(),
                'A375_diff_mean': d.A375_diff_mean.mean()}
    df_Sanson_controls = df_Sanson_controls.append(data_list, ignore_index = True)

#############################################
# Load in my HL-60 data
#############################################
df = pd.read_csv('../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_growth_sgRNA_means.csv')

df = df[['gene', 'log2fold_diff_mean', 'sgRNA']]
# df.columns = ['gene', 'HL60_diff_mean', 'sgRNA']
df = df[df.gene != 'CONTROL']

# mean across sgRNA (3 per gene, Set A)
df_avg = pd.DataFrame()
for g, d in df.groupby('gene'):
    data_list = {'gene' : g,
                'HL60_diff_mean': d.log2fold_diff_mean.mean()}
    df_avg = df_avg.append(data_list, ignore_index = True)

# control sgRNAs
df_controls = pd.read_csv('../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_growth_sgRNA_means.csv')

df_controls = df_controls[['gene', 'log2fold_diff_mean', 'sgRNA']]
df_controls.columns = ['gene', 'HL60_diff_mean', 'sgRNA']
df_controls = df_controls[df_controls.gene == 'CONTROL']

#############################################
# Generate combined dataframe with both data sources
#############################################
df_comb = pd.merge(df_avg, df_Sanson_avg, on = 'gene')
df_comb_controls = pd.merge(df_controls, df_Sanson_controls, on = 'sgRNA')

fig = plt.figure(figsize=(8,5))

gs = GridSpec(nrows=2, ncols=3 , height_ratios=[0.2, 1], width_ratios=[1, 1, 0.2])

ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[1,1])
HL60_marg = fig.add_subplot(gs[1,2])
HT29_marg = fig.add_subplot(gs[0,0])
A375_marg = fig.add_subplot(gs[0,1])


ax1.scatter(df_comb.HT29_diff_mean, df_comb.HL60_diff_mean,
          edgecolors = 'k', linewidths = 0.2, zorder = 5,
           color = '#5C82BF', s = 10, alpha = 0.2) #'#738FC1'
             # alpha = 0.15, s = 5, zorder=10)
print(np.max(df_comb.HL60_diff_mean))
ax1.scatter(df_comb_controls.HT29_diff_mean, df_comb_controls.HL60_diff_mean,
          edgecolors = 'k', linewidths = 0.2, zorder = 5,
           color = '#B8BABC', s = 10, alpha = 0.8)

ax1.set_xlabel(r'normalized log$_2$ fold-change'
                 '\n(HT29 cell line)', fontsize = 12)
ax1.set_ylabel(r'normalized log$_2$ fold-change'
                 '\n(HL-60 cell line)', fontsize = 12)
ax1.text(x = 0.25, y = -2, s = 'Pearson\ncorrelation\n' + r'$\rho$ = '
           + str(np.round(stats.pearsonr(df_comb.HT29_diff_mean,
                                         df_comb.HL60_diff_mean)[0],2)),
        fontsize = 12)
ax1.set_xlim(-3.5,2)
ax1.set_ylim(-2.3, 0.5)
ax1.hlines(0, -3.6, 2, linestyles = '--')
ax1.vlines(0, -2.3,0.5, linestyles = '--')


sns.distplot(df_comb.HL60_diff_mean, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = HL60_marg, vertical=True)

sns.distplot(df_comb.HT29_diff_mean, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = HT29_marg, vertical=False)

sns.distplot(df_comb.A375_diff_mean, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = A375_marg, vertical=False)


ax2.scatter(df_comb.A375_diff_mean, df_comb.HL60_diff_mean,
          edgecolors = 'k', linewidths = 0.2, zorder = 5,
           color = '#5C82BF', s = 10, alpha = 0.2)
             # alpha = 0.15, s = 5, zorder=10)

ax2.scatter(df_comb_controls.A375_diff_mean, df_comb_controls.HL60_diff_mean,
          edgecolors = 'k', linewidths = 0.2, zorder = 5,
           color = '#B8BABC', s = 10, alpha = 0.8)

ax2.set_xlabel(r'normalized log$_2$ fold-change'
                 '\n(A375 cell line)', fontsize = 12)
ax2.text(x = 0.25, y = -2, s = 'Pearson\ncorrelation\n' + r'$\rho$ = '
           + str(np.round(stats.pearsonr(df_comb.A375_diff_mean,
                                         df_comb.HL60_diff_mean)[0],2)),
        fontsize = 12)
ax2.set_xlim(-3.5,2)
ax2.set_ylim(-2.3,0.5)
ax2.set_yticklabels([])
ax2.hlines(0, -3.6, 2, linestyles = '--')
ax2.vlines(0, -2.3,0.5, linestyles = '--')
# ax2.spines['left'].set_position(('data', 0))
# ax2.spines['bottom'].set_position(('data', 0))

for  ax_ in [HL60_marg, HT29_marg, A375_marg]:
    ax_.get_xaxis().set_visible(False)
    ax_.get_yaxis().set_visible(False)
    ax_.set_xticklabels([])
    ax_.set_yticklabels([])

for  ax_ in [HL60_marg]:
    ax_.set_ylim(-2.3,0.5)
for  ax_ in [HT29_marg, A375_marg]:
    ax_.set_xlim(-3.5,2)


plt.tight_layout()

fig.savefig('../../figures/FigS2_prolif.pdf', bbox_inches='tight')
