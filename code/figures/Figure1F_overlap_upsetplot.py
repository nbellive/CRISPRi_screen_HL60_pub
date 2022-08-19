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

import upsetplot
from upsetplot import UpSet

plt.style.use('styleNB.mplstyle')


###############################
# upset plot - venn type diagram
###############################


def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

# calculate fdr and identify genes with fdr < 0.05
df_growth = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_growth_gene_pvalues.csv')
df_diff = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_differentiation_gene_pvalues.csv')
df_mig = pd.read_csv('../../data/screen_summary/stats/gene_avg/20211222_screen_log2fold_diffs_migration_combined_gene_pvalues.csv')

df = df_growth
df = df.append(df_diff,
        ignore_index = True)
df = df.append(df_mig,
        ignore_index = True)

# pd.read_csv('../../data/screen_summary/collated_screen_data_gene_pvalues_20210830.csv')
# df = df[df.exp.isin(['growth', 'differentiation', 'ECM_all', 'ECM_all_truncatemerge', 'ECM_fibrin_goodcollagen',
# 'ECM_fibrinonly'])]
# # df = pd.append(pd.read_csv('../../data/screen_summary/collated_screen_data_gene_transwellcombined_pvalues_20211222.csv'),
# #             ignore_index = True)
# # df = df.append(pd.read_csv('../../data/screen_summary/collated_screen_data_gene_transwellcombined_10grad_pvalues_20211222.csv'),
# #               ignore_index = True)
# # df = df.append(pd.read_csv('../../data/screen_summary/collated_screen_data_gene_transwellcombined_NOgrad_pvalues_20211222.csv'),
# #               ignore_index = True)
# df = df.append(pd.read_csv('../../data/screen_summary/collated_screen_data_gene_pvalues_20210830.csv'),
#                 ignore_index = True)

# df = df[df.exp.isin([, 'differentiation', 'growth' ,
#  'transwell_all_10grad_data', 'transwell_all_NOgrad_data'])]

# Load the compiled data sets

# df_sig = pd.DataFrame([])
# for exp, d in df.groupby('exp'):
#     d['fdr'] = fdr(d.pvalue)
#     if 'ECM' in exp:
#         d = d[d.fdr <= 0.2]
#     else:
#         d = d[d.fdr <= 0.05]
#     df_sig = df_sig.append(d, ignore_index = True)
#
# # print(df_sig.exp.unique())
# df_sig = df_sig[['gene', 'exp']]
# df_sig_mig = df_sig[df_sig['exp'].str.contains('ECM')]
# df_sig_mig = df_sig_mig.append(df_sig[df_sig.exp.str.contains('transwell')])
# df_sig_mig = df_sig_mig[['gene']].drop_duplicates()
# df_sig_mig['exp'] = 'cell migration'
# df_sig = df_sig[~df_sig['exp'].str.contains('ECM')]
# df_sig = df_sig[~df_sig['exp']str.contains('transwell')]
#
# df_sig = df_sig.append(df_sig_mig, ignore_index = True)

df_sig = pd.DataFrame([])
for exp, d in df.groupby('exp'):
    d['fdr'] = fdr(d.pvalue)
    d = d[d.fdr <= 0.05]
    df_sig = df_sig.append(d, ignore_index = True)

set_df = pd.DataFrame([])
for  sg in df_sig['gene'].unique():
    d = df_sig[df_sig['gene'] == sg]['exp'].unique()
    d = ', '.join(list(d))
    growth, differentiation, migration = False, False, False
    if 'growth' in d:
        growth = True
    if 'differentiation' in d:
        differentiation = True
    if 'migration' in d:
        migration = True
    set_df = set_df.append({'growth': growth,
                            'differentiation': differentiation,
                            'cell migration': migration,
                            'gene': 1},
                            ignore_index=True)


#%%
set_df.set_index([g for g in set_df.keys() if g != 'gene'], inplace=True)
#%%



upset = UpSet(set_df, sum_over='gene', show_counts=False,
                sort_by='cardinality',element_size=45,
                facecolor='#E3DCD1', other_dots_color='#C6C6C6',
                shading_color = '#F2F2F2')
upset.style_subsets(present = 'growth',
                edgecolor = 'k', linewidth=1)
upset.style_subsets(present = 'differentiation',
                edgecolor = 'k', linewidth=1)
upset.style_subsets(present = 'cell migration',
                edgecolor = 'k', linewidth=1)
upset.plot()

# reduce legend size:
params = {'legend.fontsize': 10,
          'ytick.labelsize': 12,
          'xtick.labelsize': 12,
          'font.size': 12}
with plt.rc_context(params):
    upset.plot()
# g = upsetplot.plot(set_df, sum_over='gene', show_counts=True,
#                 sort_by='cardinality',element_size=45,
#                 facecolor='#E3DCD1', other_dots_color='#C6C6C6',
#                 shading_color = '#F2F2F2')


# g

plt.savefig('../../figures/Fig1F_screens_upset.pdf')
