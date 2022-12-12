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

def fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

###############################
# upset plot - venn type diagram
###############################

# load in the screen data
# growth and differentiation screens
df_growth = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_growth_means_pvalue.csv')
df_diff = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')
# migration screens
df_mig1 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_0_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_chemokinesis'
df_mig2 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_10_all_means_pvalue.csv')
df_mig2['exp'] = 'migration_chemotaxis'
df_mig3 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3D'

# Append all DataFrames together
df = df_growth
df = df.append(df_diff,
        ignore_index = True)
df = df.append(df_mig1,
        ignore_index = True)
df = df.append(df_mig2,
        ignore_index = True)
df = df.append(df_mig3,
        ignore_index = True)
df = df[~df.gene.str.contains('CONTROL')]


# calculate fdr and identify genes with fdr < 0.05
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

plt.savefig('../../figures/Fig1F_screens_upset_.pdf')
