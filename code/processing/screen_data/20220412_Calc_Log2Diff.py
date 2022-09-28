import numpy as np
import pandas as pd
import gzip
import glob
import os
import sys

from sklearn import preprocessing
import matplotlib.pyplot as plt

sys.path.append('../../..')

def scale_data(array, means,stds):
    return (array-means)/stds


#################################
#################################
# The aim of this file is to convert the raw sequence counts
# into log2-foldchange data for each experiment.

# Experiments:
# 1. uHL-60 proliferation
# 2. dHL-60 differentiation
# 3. Cell migration
# - Chemotaxis (10% FBS gradient), track-etch membrane, ~2 hr time point
# - Chemotaxis (10% FBS gradient), track-etch membrane, ~6 hr time point
# - Chemokinesis (uniform 10% FBS), track-etch membrane, ~2 hr time point
# - Chemokinesis (uniform 10% FBS), track-etch membrane, ~6 hr time point
# - Amoeboid 3D (collagen/fibrin), 9 hr time point
#################################
#################################

#################################
#################################
# Note about loading in the count data:
# Not all files will have counts for every sgRNA.
# When I merge, in order to avoid losing
# counts, those that are missing will be given
# a value of 0 counts.

# Also note that for cell migration
# screen data, the individual experiments
# tend to be noisier. To better handle this,
# the log2fold-change values are renormalized
# such that the control sgRNAs follow
# a normal distribution.
#################################
#################################
#
# #################################
# #################################
# # Growth/proliferation data
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
# df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']
#
# df = df_IDs.copy()
# exp_list_set1 = ['NB101', 'NB102', 'NB113', 'NB114']
#
# for exp in exp_list_set1:
#     df_temp = pd.read_csv('../../../data/seq/20210210_novogene/raw_data/' +
#                           exp + '/test_trim_seqtk_sgRNAcounts_20210218.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# exp_list_set2 = ['NB201', 'NB202', 'NB203', 'NB204']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210810_novogene/raw_data/'
#
# for exp in exp_list_set2:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210810.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# # from first screen exps
# df['NB113_NB101_diff'] = np.log2(df.counts_NB113 + 32) - np.log2(df.counts_NB113.median()) - (np.log2(df.counts_NB101 + 32) - np.log2(df.counts_NB101.median()))
# df['NB114_NB102_diff'] = np.log2(df.counts_NB114 + 32) - np.log2(df.counts_NB114.median()) - (np.log2(df.counts_NB102 + 32) - np.log2(df.counts_NB102.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB113_NB101_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# # df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# df_temp['exp'] = 'growth_rep1_A'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB114_NB102_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# # df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# df_temp['exp'] = 'growth_rep1_B'
# df_compare = df_compare.append(df_temp)
#
#
# # from second screen exps
# df['NB203_NB201_diff'] = np.log2(df.counts_NB203 + 32) - np.log2(df.counts_NB203.median()) - (np.log2(df.counts_NB201 + 32) - np.log2(df.counts_NB201.median()))
# df['NB204_NB202_diff'] = np.log2(df.counts_NB204 + 32) - np.log2(df.counts_NB204.median()) - (np.log2(df.counts_NB202 + 32) - np.log2(df.counts_NB202.median()))
#
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB203_NB201_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# # df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# df_temp['exp'] = 'growth_rep2_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB204_NB202_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# # df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# df_temp['exp'] = 'growth_rep2_B'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_growth_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_growth_sgRNA_means.csv')
#
#
# #################################
# #################################
# # Differentiation data
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
# df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']
#
# df = df_IDs.copy()
# exp_list_set1 = ['NB101', 'NB102', 'NB103', 'NB104', 'NB112', 'NB113', 'NB114',
#             'NB118', 'NB119']
#
# for exp in exp_list_set1:
#     df_temp = pd.read_csv('../../../data/seq/20210210_novogene/raw_data/' +
#                           exp + '/test_trim_seqtk_sgRNAcounts_20210218.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# exp_list_set2 = ['NB115', 'NB116']
#
# for exp in exp_list_set2:
#     df_temp = pd.read_csv('../../../data/seq/20210310_novogene/raw_data/' +
#                           exp + '/trim_seqtk_sgRNAcounts_20210312_truncate.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# exp_list_set3 = ['NB139', 'NB140']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210614_novogene/raw_data/'
#
# for exp in exp_list_set3:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210714.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# exp_list_set4 = ['NB201', 'NB202', 'NB203', 'NB204', 'NB205', 'NB206', 'NB224', 'NB225']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210810_novogene/raw_data/'
#
# for exp in exp_list_set4:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210810.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# df['NB103_NB101_diff'] = np.log2(df.counts_NB103 + 32) - np.log2(df.counts_NB103.median()) - (np.log2(df.counts_NB101 + 32) - np.log2(df.counts_NB101.median()))
# df['NB104_NB102_diff'] = np.log2(df.counts_NB104 + 32) - np.log2(df.counts_NB104.median()) - (np.log2(df.counts_NB102 + 32) - np.log2(df.counts_NB102.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB103_NB101_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB103_rep1_A'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB104_NB102_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB104_rep1_B'
# df_compare = df_compare.append(df_temp)
#
# df['NB139_NB113_diff'] = np.log2(df.counts_NB139 + 32) - np.log2(df.counts_NB139.median()) - (np.log2(df.counts_NB113 + 32) - np.log2(df.counts_NB113.median()))
# df['NB140_NB114_diff'] = np.log2(df.counts_NB140 + 32) - np.log2(df.counts_NB140.median()) - (np.log2(df.counts_NB114 + 32) - np.log2(df.counts_NB114.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB139_NB113_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB139_rep2_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB140_NB114_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB140_rep2_B'
# df_compare = df_compare.append(df_temp)
#
# df['NB205_NB201_diff'] = np.log2(df.counts_NB205 + 32) - np.log2(df.counts_NB205.median()) - (np.log2(df.counts_NB201 + 32) - np.log2(df.counts_NB201.median()))
# df['NB206_NB202_diff'] = np.log2(df.counts_NB206 + 32) - np.log2(df.counts_NB206.median()) - (np.log2(df.counts_NB202 + 32) - np.log2(df.counts_NB202.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB205_NB201_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB103_rep3_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB206_NB202_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB104_rep3_B'
# df_compare = df_compare.append(df_temp)
#
# df['NB224_NB203_diff'] = np.log2(df.counts_NB224 + 32) - np.log2(df.counts_NB224.median()) - (np.log2(df.counts_NB203 + 32) - np.log2(df.counts_NB203.median()))
# df['NB225_NB204_diff'] = np.log2(df.counts_NB225 + 32) - np.log2(df.counts_NB225.median()) - (np.log2(df.counts_NB204 + 32) - np.log2(df.counts_NB204.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB224_NB203_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB139_rep4_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB225_NB204_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['exp'] = 'NB140_rep4_B'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_differentiation_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_differentiation_sgRNA_means.csv')
#
#
#
#
#
#
# #################################
# #################################
# # Cell migration - track-etch membrane, chemokinesis (uniform 10% FBS), ~6 hr
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
# df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']
#
# df = df_IDs.copy()
# exp_list_set1 = ['NB101', 'NB102', 'NB103', 'NB104', 'NB112', 'NB113', 'NB114',
#             'NB118', 'NB119']
#
# for exp in exp_list_set1:
#     df_temp = pd.read_csv('../../../data/seq/20210210_novogene/raw_data/' +
#                           exp + '/test_trim_seqtk_sgRNAcounts_20210218.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# exp_list_set2 = ['NB105', 'NB106', 'NB107', 'NB108', 'NB109', 'NB110', 'NB111',
#                   'NB115', 'NB116', 'NB117', 'NB120']
#
#
# for exp in exp_list_set2:
#     df_temp = pd.read_csv('../../../data/seq/20210310_novogene/raw_data/' +
#                           exp + '/trim_seqtk_sgRNAcounts_20210312_truncate.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# exp_list_set3 = ['NB121', 'NB122', 'NB125', 'NB126', 'NB129', 'NB130']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210614_novogene/raw_data/'
#
# for exp in exp_list_set3:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210714.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# # top
# # A
# df['NB125_NB121_diff'] = np.log2(df.counts_NB125 + 32) - np.log2(df.counts_NB125.median()) - (np.log2(df.counts_NB121 + 32) - np.log2(df.counts_NB121.median()))
# # B
# df['NB129_NB122_diff'] = np.log2(df.counts_NB129 + 32) - np.log2(df.counts_NB129.median()) - (np.log2(df.counts_NB122 + 32) - np.log2(df.counts_NB122.median()))
#
# # bottom
# # A
# df['NB126_NB121_diff'] = np.log2(df.counts_NB126 + 32) - np.log2(df.counts_NB126.median()) - (np.log2(df.counts_NB121 + 32) - np.log2(df.counts_NB121.median()))
# # B
# df['NB130_NB122_diff'] = np.log2(df.counts_NB130 + 32) - np.log2(df.counts_NB130.median()) - (np.log2(df.counts_NB122 + 32) - np.log2(df.counts_NB122.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB125_NB121_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_0_A_1'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB126_NB121_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_0_A_1'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB129_NB122_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_0_A_2'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB130_NB122_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_0_A_2'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_0_6hr_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_0_6hr_sgRNA_means.csv')
#
# #################################
# #################################
# # Cell migration - track-etch membrane, chemotaxis (10% FBS gradient), ~6 hr,
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# # load in the data from the second round of experiments (already loaded in some data above)
# # 22 um
# exp_list_set_22 = ['NB213', 'NB214', 'NB207', 'NB208', 'NB209', 'NB210']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210810_novogene/raw_data/'
#
# for exp in exp_list_set_22:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210810.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# exp_list_set_10 = ['NB221', 'NB222', 'NB223', 'NB228', 'NB229', 'NB230']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210810_novogene/raw_data/'
#
# for exp in exp_list_set_10:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210810.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# df['NB105_NB103_diff'] = np.log2(df.counts_NB105 + 32) - np.log2(df.counts_NB105.median()) - (np.log2(df.counts_NB103 + 32) - np.log2(df.counts_NB103.median()))
# df['NB106_NB103_diff'] = np.log2(df.counts_NB106 + 32) - np.log2(df.counts_NB106.median()) - (np.log2(df.counts_NB103 + 32) - np.log2(df.counts_NB103.median()))
# df['NB107_NB103_diff'] = np.log2(df.counts_NB107 + 32) - np.log2(df.counts_NB107.median()) - (np.log2(df.counts_NB103 + 32) - np.log2(df.counts_NB103.median()))
# df['NB108_NB103_diff'] = np.log2(df.counts_NB108 + 32) - np.log2(df.counts_NB108.median()) - (np.log2(df.counts_NB103 + 32) - np.log2(df.counts_NB103.median()))
#
# df['NB109_NB104_diff'] = np.log2(df.counts_NB109 + 32) - np.log2(df.counts_NB109.median()) - (np.log2(df.counts_NB104 + 32) - np.log2(df.counts_NB104.median()))
# df['NB110_NB104_diff'] = np.log2(df.counts_NB110 + 32) - np.log2(df.counts_NB110.median()) - (np.log2(df.counts_NB104 + 32) - np.log2(df.counts_NB104.median()))
# df['NB111_NB104_diff'] = np.log2(df.counts_NB111 + 32) - np.log2(df.counts_NB111.median()) - (np.log2(df.counts_NB104 + 32) - np.log2(df.counts_NB104.median()))
# df['NB112_NB104_diff'] = np.log2(df.counts_NB112 + 32) - np.log2(df.counts_NB112.median()) - (np.log2(df.counts_NB104 + 32) - np.log2(df.counts_NB104.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB105_NB103_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_A_1'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB106_NB103_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_A_1'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB107_NB103_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_A_2'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB108_NB103_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_A_2'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB109_NB104_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_B_1'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB110_NB104_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_B_1'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB111_NB104_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_B_2'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB112_NB104_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_B_2'
# df_compare = df_compare.append(df_temp)
#
#
# ####### Set from second experiments
# # 22 um exps
# # top
# # A
# df['NB207_NB213_diff'] = np.log2(df.counts_NB207 + 32) - np.log2(df.counts_NB207.median()) - (np.log2(df.counts_NB213 + 32) - np.log2(df.counts_NB213.median()))
# # B
# df['NB209_NB214_diff'] = np.log2(df.counts_NB209 + 32) - np.log2(df.counts_NB209.median()) - (np.log2(df.counts_NB214 + 32) - np.log2(df.counts_NB214.median()))
#
# # bottom
# # A
# df['NB208_NB213_diff'] = np.log2(df.counts_NB208 + 32) - np.log2(df.counts_NB208.median()) - (np.log2(df.counts_NB213 + 32) - np.log2(df.counts_NB213.median()))
# # B
# df['NB210_NB214_diff'] = np.log2(df.counts_NB210 + 32) - np.log2(df.counts_NB210.median()) - (np.log2(df.counts_NB214 + 32) - np.log2(df.counts_NB214.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB207_NB213_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_A_22um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB208_NB213_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_A_22um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB209_NB214_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_B_22um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB210_NB214_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_B_22um'
# df_compare = df_compare.append(df_temp)
#
# #######
# # 22 um exps
#
# # top
# # A
# df['NB222_NB221_diff'] = np.log2(df.counts_NB222 + 32) - np.log2(df.counts_NB222.median()) - (np.log2(df.counts_NB221 + 32) - np.log2(df.counts_NB221.median()))
# # B
# df['NB229_NB228_diff'] = np.log2(df.counts_NB229 + 32) - np.log2(df.counts_NB229.median()) - (np.log2(df.counts_NB228 + 32) - np.log2(df.counts_NB228.median()))
#
# # bottom
# # A
# df['NB223_NB221_diff'] = np.log2(df.counts_NB223 + 32) - np.log2(df.counts_NB223.median()) - (np.log2(df.counts_NB221 + 32) - np.log2(df.counts_NB221.median()))
# # B
# df['NB230_NB228_diff'] = np.log2(df.counts_NB230 + 32) - np.log2(df.counts_NB230.median()) - (np.log2(df.counts_NB228 + 32) - np.log2(df.counts_NB228.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB222_NB221_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_A_10um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB223_NB221_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_A_10um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB229_NB228_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_10_B_10um'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB230_NB228_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_10_B_10um'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_10_6hr_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_10_6hr_sgRNA_means.csv')
#
#
# #################################
# #################################
# # Cell migration - track-etch membrane, chemokinesis (uniform 10% FBS), ~2 hr,
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
# df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']
#
# df = df_IDs.copy()
# exp_list_set1 = ['NB123', 'NB124', 'NB127', 'NB131', 'NB128', 'NB132', 'NB133', 'NB134', 'NB135', 'NB136']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210614_novogene/raw_data/'
#
# for exp in exp_list_set1:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210714.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# # 2 hrs 0 FBS gradient
# # top
# # A
# df['NB127_NB123_diff'] = np.log2(df.counts_NB127 + 32) - np.log2(df.counts_NB127.median()) - (np.log2(df.counts_NB123 + 32) - np.log2(df.counts_NB123.median()))
# # B
# df['NB128_NB124_diff'] = np.log2(df.counts_NB128 + 32) - np.log2(df.counts_NB128.median()) - (np.log2(df.counts_NB124 + 32) - np.log2(df.counts_NB124.median()))
#
# # bottom
# # A
# df['NB131_NB123_diff'] = np.log2(df.counts_NB131 + 32) - np.log2(df.counts_NB131.median()) - (np.log2(df.counts_NB123 + 32) - np.log2(df.counts_NB123.median()))
# # B
# df['NB132_NB124_diff'] = np.log2(df.counts_NB132 + 32) - np.log2(df.counts_NB132.median()) - (np.log2(df.counts_NB124 + 32) - np.log2(df.counts_NB124.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB127_NB123_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_2hr_0_A'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB128_NB124_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_2hr_0_B'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB131_NB123_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_2hr_0_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB132_NB124_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_2hr_0_B'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_0_2hr_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_0_2hr_sgRNA_means.csv')
#
#
# #################################
# #################################
# # Cell migration - track-etch membrane, chemotaxis (10% FBS gradient), ~2 hr,
# #################################
# #################################
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# # top
# # A
# df['NB133_NB123_diff'] = np.log2(df.counts_NB133 + 32) - np.log2(df.counts_NB133.median()) - (np.log2(df.counts_NB123 + 32) - np.log2(df.counts_NB123.median()))
# # B
# df['NB135_NB124_diff'] = np.log2(df.counts_NB135 + 32) - np.log2(df.counts_NB135.median()) - (np.log2(df.counts_NB124 + 32) - np.log2(df.counts_NB124.median()))
#
# # bottom
# # A
# df['NB134_NB123_diff'] = np.log2(df.counts_NB134 + 32) - np.log2(df.counts_NB134.median()) - (np.log2(df.counts_NB123 + 32) - np.log2(df.counts_NB123.median()))
# # B
# df['NB136_NB124_diff'] = np.log2(df.counts_NB136 + 32) - np.log2(df.counts_NB136.median()) - (np.log2(df.counts_NB124 + 32) - np.log2(df.counts_NB124.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB133_NB123_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_2hr_10_A'
# df_compare = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB135_NB124_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'top_2hr_10_B'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB134_NB123_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_2hr_10_A'
# df_compare = df_compare.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB136_NB124_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'bottom_2hr_10_B'
# df_compare = df_compare.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_10_2hr_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220412_screen_log2fold_diffs_tracketch_10_2hr_sgRNA_means.csv')


# #################################
# #################################
# # Cell migration - Amoeboid 3D (collagen/fibrin), uniform 10% FBS
# #################################
# #################################
#
# #################################
# # load in sgRNA counts
# #################################
# # load in sgRNA library IDs
# df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
# df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']
#
# df = df_IDs.copy()
# exp_list_set1 = ['NB101', 'NB102', 'NB103', 'NB104', 'NB112', 'NB113', 'NB114',
#             'NB118', 'NB119']
#
# for exp in exp_list_set1:
#     df_temp = pd.read_csv('../../../data/seq/20210210_novogene/raw_data/' +
#                           exp + '/test_trim_seqtk_sgRNAcounts_20210218.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# exp_list_set2 = ['NB105', 'NB106', 'NB107', 'NB108', 'NB109', 'NB110', 'NB111',
#                   'NB115', 'NB116', 'NB117', 'NB120']
#
# for exp in exp_list_set2:
#     df_temp = pd.read_csv('../../../data/seq/20210310_novogene/raw_data/' +
#                           exp + '/trim_seqtk_sgRNAcounts_20210312_truncate.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
#
# exp_list_set3 = ['NB137', 'NB138', 'NB139', 'NB140', 'NB141', 'NB142',
#                  'NB143', 'NB144', 'NB145', 'NB146', 'NB147']
# dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210614_novogene/raw_data/'
#
# # a bit of count threshold filtered based on some observations with the data
# count_threshold = {'NB137' : 0,
#                    'NB138' : 0,
#                    'NB139' : 0,
#                    'NB140' : 0,
#                    'NB141' : 50,
#                    'NB142' : 40,
#                    'NB143' : 60,
#                    'NB144' : 30,
#                    'NB146' : 30,
#                    'NB145' : 30,
#                    'NB147' : 70
#                   }
#
# for exp in exp_list_set3:
#     df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210714.csv')
#
#     df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
#     df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
#     df_temp_notmatched[''.join(['counts_',exp])] = 0
#
#     df_temp_matched = df_temp_matched[['sgRNA', 'index']]
#
#     df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
#
#     df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])
#
#     df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')
#
# #################################
# # Calculate log2 fold change (with median normalization and 'prior' of log2(32))
# #################################
# # fibrin
# df['NB117_NB115_diff'] = np.log2(df.counts_NB117 + 32) - np.log2(df.counts_NB117.median()) - (np.log2(df.counts_NB115 + 32) - np.log2(df.counts_NB115.median()))
# df['NB119_NB116_diff'] = np.log2(df.counts_NB119 + 32) - np.log2(df.counts_NB119.median()) - (np.log2(df.counts_NB116 + 32) - np.log2(df.counts_NB116.median()))
#
# # collagen
# df['NB118_NB115_diff'] = np.log2(df.counts_NB118 + 32) - np.log2(df.counts_NB118.median()) - (np.log2(df.counts_NB115 + 32) - np.log2(df.counts_NB115.median()))
# df['NB120_NB116_diff'] = np.log2(df.counts_NB120 + 32) - np.log2(df.counts_NB120.median()) - (np.log2(df.counts_NB116 + 32) - np.log2(df.counts_NB116.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB117_NB115_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'fibrin_1_A'
# df_compare_ECM = df_temp.copy()
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB119_NB116_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'fibrin_1_B'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB118_NB115_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'collagen_1_A'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB120_NB116_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'collagen_1_B'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
#
# # Replicate 2
# # fibrin
# df['NB141_NB137_diff'] = np.log2(df.counts_NB141 + 32) - np.log2(df.counts_NB141.median()) - (np.log2(df.counts_NB137 + 32) - np.log2(df.counts_NB137.median()))
#
# # collagen
# df['NB142_NB137_diff'] = np.log2(df.counts_NB142 + 32) - np.log2(df.counts_NB142.median()) - (np.log2(df.counts_NB137 + 32) - np.log2(df.counts_NB137.median()))
# df['NB143_NB138_diff'] = np.log2(df.counts_NB143 + 32) - np.log2(df.counts_NB143.median()) - (np.log2(df.counts_NB138 + 32) - np.log2(df.counts_NB138.median()))
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB141_NB137_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'fibrin_2_A'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
# # Replicate 3
#
# # fibrin
# df['NB144_NB139_diff'] = np.log2(df.counts_NB144 + 32) - np.log2(df.counts_NB144.median()) - (np.log2(df.counts_NB139 + 32) - np.log2(df.counts_NB139.median()))
# df['NB146_NB140_diff'] = np.log2(df.counts_NB146 + 32) - np.log2(df.counts_NB146.median()) - (np.log2(df.counts_NB140 + 32) - np.log2(df.counts_NB140.median()))
#
# # collagen
# df['NB145_NB139_diff'] = np.log2(df.counts_NB145 + 32) - np.log2(df.counts_NB145.median()) - (np.log2(df.counts_NB139 + 32) - np.log2(df.counts_NB139.median()))
# df['NB147_NB140_diff'] = np.log2(df.counts_NB147 + 32) - np.log2(df.counts_NB147.median()) - (np.log2(df.counts_NB140 + 32) - np.log2(df.counts_NB140.median()))
#
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB144_NB139_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'fibrin_3_A'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
# df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB146_NB140_diff']]
# df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
# s = preprocessing.StandardScaler()
# s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
# median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
# scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
# df_temp['diff'] = scale_new_data
# df_temp['exp'] = 'fibrin_3_B'
# df_compare_ECM = df_compare_ECM.append(df_temp)
#
# #################################
# # Save collated log2fold change values to disk
# #################################
# df_compare_ECM.to_csv('../../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_all.csv')
#
# #################################
# # Calculate mean values for each sgRNA and save to disk
# #################################
# df_mean = pd.DataFrame()
# for sg, d in df_compare_ECM.groupby('sgRNA'):
#     data_list = {'sgRNA' : sg,
#     'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
#     'diff' : d['diff'].mean()}
#
#     df_mean = df_mean.append(data_list, ignore_index = True)
#
# df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]
#
# #cleanup dataframe columns
# df_mean = df_mean[['gene', 'sgRNA', 'diff']]
# df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']
#
# df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_sgRNA_means.csv')

#################################
#################################
# Cell migration - Amoeboid 3D (collagen/fibrin), uniform 10% FBS
#################################
#################################

#################################
# load in sgRNA counts
#################################
# load in sgRNA library IDs
df_IDs = pd.read_csv('../../../data/seq/Sanson_Dolc_CRISPRI_IDs_libA.csv')
df_IDs.columns = ['sgRNA', 'annotated_gene_symbol', 'annotated_gene_ID']

df = df_IDs.copy()
exp_list_set1 = ['NB101', 'NB102', 'NB103', 'NB104', 'NB112', 'NB113', 'NB114',
            'NB118', 'NB119']

for exp in exp_list_set1:
    df_temp = pd.read_csv('../../../data/seq/20210210_novogene/raw_data/' +
                          exp + '/test_trim_seqtk_sgRNAcounts_20210218.csv')

    df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
    df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
    df_temp_notmatched[''.join(['counts_',exp])] = 0

    df_temp_matched = df_temp_matched[['sgRNA', 'index']]

    df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
    df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])

    df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')


exp_list_set2 = ['NB105', 'NB106', 'NB107', 'NB108', 'NB109', 'NB110', 'NB111',
                  'NB115', 'NB116', 'NB117', 'NB120']

for exp in exp_list_set2:
    df_temp = pd.read_csv('../../../data/seq/20210310_novogene/raw_data/' +
                          exp + '/trim_seqtk_sgRNAcounts_20210312_truncate.csv')

    df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
    df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
    df_temp_notmatched[''.join(['counts_',exp])] = 0

    df_temp_matched = df_temp_matched[['sgRNA', 'index']]

    df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]
    df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])

    df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')


exp_list_set3 = ['NB137', 'NB138', 'NB139', 'NB140', 'NB141', 'NB142',
                 'NB143', 'NB144', 'NB145', 'NB146', 'NB147']
dirname = '/Volumes/Belliveau_RAW_3_JTgroup/Sequencing_RAW/20210614_novogene/raw_data/'

# a bit of count threshold filtered based on some observations with the data
count_threshold = {'NB137' : 0,
                   'NB138' : 0,
                   'NB139' : 0,
                   'NB140' : 0,
                   'NB141' : 50,
                   'NB142' : 40,
                   'NB143' : 60,
                   'NB144' : 30,
                   'NB146' : 30,
                   'NB145' : 30,
                   'NB147' : 70
                  }

for exp in exp_list_set3:
    df_temp = pd.read_csv(dirname + exp + '/trim_seqtk_sgRNAcounts_20210714.csv')

    df_temp_matched = df_temp[df_temp.sgRNA.isin(df_IDs.sgRNA.values)]
    df_temp_notmatched = df_IDs[~df_IDs.sgRNA.isin(df_temp_matched.sgRNA.values)]
    df_temp_notmatched[''.join(['counts_',exp])] = 0

    df_temp_matched = df_temp_matched[['sgRNA', 'index']]

    df_temp_matched.columns = ['sgRNA', ''.join(['counts_',exp])]

    df_temp_matched = df_temp_matched.append(df_temp_notmatched[['sgRNA', ''.join(['counts_',exp])]])

    df = pd.merge(df, df_temp_matched[['sgRNA',''.join(['counts_',exp])]], on = 'sgRNA')

#################################
# Calculate log2 fold change (with median normalization and 'prior' of log2(32))
#################################
# fibrin
df['NB117_NB115_diff'] = np.log2(df.counts_NB117 + 32) - np.log2(df.counts_NB117.median()) - (np.log2(df.counts_NB115 + 32) - np.log2(df.counts_NB115.median()))
df['NB119_NB116_diff'] = np.log2(df.counts_NB119 + 32) - np.log2(df.counts_NB119.median()) - (np.log2(df.counts_NB116 + 32) - np.log2(df.counts_NB116.median()))

# collagen
df['NB118_NB115_diff'] = np.log2(df.counts_NB118 + 32) - np.log2(df.counts_NB118.median()) - (np.log2(df.counts_NB115 + 32) - np.log2(df.counts_NB115.median()))
df['NB120_NB116_diff'] = np.log2(df.counts_NB120 + 32) - np.log2(df.counts_NB120.median()) - (np.log2(df.counts_NB116 + 32) - np.log2(df.counts_NB116.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB117_NB115_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'fibrin_1_A'
df_compare_ECM = df_temp.copy()

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB119_NB116_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
scale_new_data = scale_data(df_temp['diff'], means = s.mean_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'fibrin_1_B'
df_compare_ECM = df_compare_ECM.append(df_temp)

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB118_NB115_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
df_temp['diff'] = -df_temp['diff']
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'collagen_1_A'
df_compare_ECM = df_compare_ECM.append(df_temp)

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB120_NB116_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(-df_temp['diff'])
df_temp['diff'] = -df_temp['diff']
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'collagen_1_B'
df_compare_ECM = df_compare_ECM.append(df_temp)


# Replicate 2
# fibrin
df['NB141_NB137_diff'] = np.log2(df.counts_NB141 + 32) - np.log2(df.counts_NB141.median()) - (np.log2(df.counts_NB137 + 32) - np.log2(df.counts_NB137.median()))

# collagen
df['NB142_NB137_diff'] = np.log2(df.counts_NB142 + 32) - np.log2(df.counts_NB142.median()) - (np.log2(df.counts_NB137 + 32) - np.log2(df.counts_NB137.median()))
df['NB143_NB138_diff'] = np.log2(df.counts_NB143 + 32) - np.log2(df.counts_NB143.median()) - (np.log2(df.counts_NB138 + 32) - np.log2(df.counts_NB138.median()))

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB141_NB137_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'fibrin_2_A'
df_compare_ECM = df_compare_ECM.append(df_temp)

# Replicate 3

# fibrin
df['NB144_NB139_diff'] = np.log2(df.counts_NB144 + 32) - np.log2(df.counts_NB144.median()) - (np.log2(df.counts_NB139 + 32) - np.log2(df.counts_NB139.median()))
df['NB146_NB140_diff'] = np.log2(df.counts_NB146 + 32) - np.log2(df.counts_NB146.median()) - (np.log2(df.counts_NB140 + 32) - np.log2(df.counts_NB140.median()))

# collagen
df['NB145_NB139_diff'] = np.log2(df.counts_NB145 + 32) - np.log2(df.counts_NB145.median()) - (np.log2(df.counts_NB139 + 32) - np.log2(df.counts_NB139.median()))
df['NB147_NB140_diff'] = np.log2(df.counts_NB147 + 32) - np.log2(df.counts_NB147.median()) - (np.log2(df.counts_NB140 + 32) - np.log2(df.counts_NB140.median()))


df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB144_NB139_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'fibrin_3_A'
df_compare_ECM = df_compare_ECM.append(df_temp)

df_temp = df[['sgRNA', 'annotated_gene_symbol', 'NB146_NB140_diff']]
df_temp.columns = ['sgRNA', 'annotated_gene_symbol', 'diff']
# df_temp['diff'] = preprocessing.scale(df_temp['diff'])
s = preprocessing.StandardScaler()
s.fit(np.array([df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].values]).T)
median_ = df_temp[df_temp.annotated_gene_symbol == 'CONTROL']['diff'].median()
scale_new_data = scale_data(df_temp['diff'], means = median_, stds = s.scale_)
df_temp['diff'] = scale_new_data
df_temp['exp'] = 'fibrin_3_B'
df_compare_ECM = df_compare_ECM.append(df_temp)

#################################
# Save collated log2fold change values to disk
#################################
df_compare_ECM.to_csv('../../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_all_4.csv')

#################################
# Calculate mean values for each sgRNA and save to disk
#################################
df_mean = pd.DataFrame()
for sg, d in df_compare_ECM.groupby('sgRNA'):
    data_list = {'sgRNA' : sg,
    'annotated_gene_symbol' : d.annotated_gene_symbol.unique(),
    'diff' : d['diff'].mean()}

    df_mean = df_mean.append(data_list, ignore_index = True)

df_mean['gene'] = [i[0] for i in df_mean.annotated_gene_symbol.values]

#cleanup dataframe columns
df_mean = df_mean[['gene', 'sgRNA', 'diff']]
df_mean.columns = ['gene', 'sgRNA', 'log2fold_diff_mean']

df_mean.to_csv('../../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_sgRNA_means_4.csv')
