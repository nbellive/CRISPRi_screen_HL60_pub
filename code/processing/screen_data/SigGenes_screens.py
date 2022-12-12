import numpy as np
import glob
import pandas as pd

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
df_growth = pd.read_csv('../../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_growth_means_pvalue.csv')
df_growth['exp'] = 'proliferation'
df_diff = pd.read_csv('../../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')
# migration screens
df_mig1 = pd.read_csv('../../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_0_all_means_pvalue.csv')
df_mig1['exp'] = 'migration_chemokinesis'
df_mig2 = pd.read_csv('../../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_tracketch_10_all_means_pvalue.csv')
df_mig2['exp'] = 'migration_chemotaxis'
df_mig3 = pd.read_csv('../../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_3D_amoeboid_means_pvalue.csv')
df_mig3['exp'] = 'migration_3Damoeboid'

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

sign_dict = {-1 : 'negative', 1: 'positive'}

# calculate fdr and identify genes with fdr < 0.05
df_sig = pd.DataFrame()
for exp, d in df.groupby('exp'):
    d['fdr'] = fdr(d.pvalue)
    d = d[d.fdr <= 0.05]

    for g, _d in d.groupby('gene'):
        data = {'gene' : g,
                'log2foldchange_sign' : sign_dict[np.sign(_d.log2fold_diff_mean.values[0])],
                'CRISPRi_screen' : exp}
        df_sig = df_sig.append(data, ignore_index = True)

df_sig.to_csv('../../../data/screen_summary/SuppTable1_ScreenSummary.csv', index = False)
