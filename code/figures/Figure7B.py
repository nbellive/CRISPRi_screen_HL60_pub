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
# identify cell migration genes  , fdr 0.05
###############################

# load in the screen data
# growth and differentiation screens
df_growth = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_growth_means_pvalue.csv')
df_diff = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')
# migration screens
df_mig1 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_tracketch'
df_mig3 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3D'

# Append all DataFrames together
df = df_growth
df = df.append(df_diff,
        ignore_index = True)
df = df.append(df_mig1,
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
                            'gene': sg},
                            ignore_index=True)


set_df = set_df[set_df['cell migration'] == True]
set_df = set_df[set_df['growth'] == False]
set_df = set_df[set_df['differentiation'] == False]

# migration screens - now grab significant genes for cell migration
df_mig1 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_tracketch'
df_mig1 = df_mig1[~df_mig1.gene.str.contains('CONTROL')]
df_mig1['fdr'] = fdr(df_mig1.pvalue)
df_mig1 = df_mig1[df_mig1.gene.isin(set_df.gene.values)].sort_values('log2fold_diff_mean')
df_mig1 = df_mig1[['gene','log2fold_diff_mean','exp','fdr']]
df_mig1.to_csv('../../figures/Fig7B_summary_migration_tracketch_exclusive_fdr05.csv', index = False)


df_mig3 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3D'
df_mig3 = df_mig3[~df_mig3.gene.str.contains('CONTROL')]
df_mig3['fdr'] = fdr(df_mig3.pvalue)
df_mig3 = df_mig3[df_mig3.gene.isin(set_df.gene.values)].sort_values('log2fold_diff_mean')
df_mig3 = df_mig3[['gene','log2fold_diff_mean','exp','fdr']]
df_mig3.to_csv('../../figures/Fig7B_summary_migration_3D_exclusive_fdr05.csv', index = False)


###############################
# identify cell migration genes  , fdr 0.05
###############################

# load in the screen data
# growth and differentiation screens
df_growth = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_growth_means_pvalue.csv')
df_diff = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')
# migration screens
df_mig1 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_tracketch'
df_mig3 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3D'

# Append all DataFrames together
df = df_growth
df = df.append(df_diff,
        ignore_index = True)
df = df.append(df_mig1,
        ignore_index = True)
df = df.append(df_mig3,
        ignore_index = True)
df = df[~df.gene.str.contains('CONTROL')]


# calculate fdr and identify genes with fdr < 0.05
df_sig = pd.DataFrame([])
for exp, d in df.groupby('exp'):
    d['fdr'] = fdr(d.pvalue)
    d = d[d.fdr <= 0.2]
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
                            'gene': sg},
                            ignore_index=True)


set_df = set_df[set_df['cell migration'] == True]
set_df = set_df[set_df['growth'] == False]
set_df = set_df[set_df['differentiation'] == False]

# migration screens - now grab significant genes for cell migration
df_mig1 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_tracketch'
df_mig1 = df_mig1[~df_mig1.gene.str.contains('CONTROL')]
df_mig1['fdr'] = fdr(df_mig1.pvalue)
df_mig1 = df_mig1[df_mig1.gene.isin(set_df.gene.values)].sort_values('log2fold_diff_mean')
df_mig1 = df_mig1[['gene','log2fold_diff_mean','exp','fdr']]
df_mig1.to_csv('../../figures/Fig7B_summary_migration_tracketch_exclusive_fdr2.csv', index = False)


df_mig3 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3D'
df_mig3 = df_mig3[~df_mig3.gene.str.contains('CONTROL')]
df_mig3['fdr'] = fdr(df_mig3.pvalue)
df_mig3 = df_mig3[df_mig3.gene.isin(set_df.gene.values)].sort_values('log2fold_diff_mean')
df_mig3 = df_mig3[['gene','log2fold_diff_mean','exp','fdr']]
df_mig3.to_csv('../../figures/Fig7B_summary_migration_3D_exclusive_fdr2.csv', index = False)
