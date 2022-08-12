import numpy as np
import pandas as pd
import random
import glob


fdir = '../../../data/screen_summary/log2foldchange/'

exps = ['20220412_screen_log2fold_diffs_growth_sgRNA_means.csv',
        '20220412_screen_log2fold_diffs_differentiation_sgRNA_means.csv',
        '20220412_screen_log2fold_diffs_tracketch_0_2hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_0_6hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_10_2hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_10_6hr_all.csv',
        '20220516_screen_log2fold_diffs_ECM_sgRNA_all.csv']

# df_collate = pd.DataFrame()
#
# #######################
# # growth screen
# df = pd.read_csv(fdir + exps[0])
# df = df[['gene', 'sgRNA', 'log2fold_diff_mean']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff']
# df['exp'] = 'growth'
#
# # assign unique number to each control sgRNA
# sgRNA_dict = dict(zip(df[df.gene == 'CONTROL'].sgRNA.unique(),
#                       np.arange(len(df[df.gene == 'CONTROL']))))
#
# df_ctrl = df[df.gene == 'CONTROL']
# df = df[df.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df = df.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # differentiation screen
# df = pd.read_csv(fdir + exps[1])
# df = df[['gene', 'sgRNA', 'log2fold_diff_mean']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff']
# df['exp'] = 'differentiation'
#
# df_ctrl = df[df.gene == 'CONTROL']
# df = df[df.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df = df.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # migration, tracketch_0_2hr
# df = pd.read_csv(fdir + exps[2])
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('top')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_0_2hr_top'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('bottom')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_0_2hr_bottom'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # migration, tracketch_0_6hr
# df = pd.read_csv(fdir + exps[3])
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('top')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_0_6hr_top'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('bottom')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_0_6hr_bottom'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # migration, tracketch_10_2hr
# df = pd.read_csv(fdir + exps[4])
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('top')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_10_2hr_top'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('bottom')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_10_2hr_bottom'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # migration, tracketch_10_2hr
# df = pd.read_csv(fdir + exps[5])
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('top')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_10_6hr_top'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('bottom')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'tracketch_10_6hr_bottom'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + exps[6])
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('fibrin')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'ECM_fibrin'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# df_temp = pd.DataFrame()
# for sg, d in df[df.exp.str.contains('collagen')].groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'ECM_collagen'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data.csv')
#

#######################
#######################
#######################
#######################

df_collate2 = pd.DataFrame()

#######################
# migration, amoeboid 3D
df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_sgRNA_all_2.csv')
df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
df_temp = pd.DataFrame()
for sg, d in df[df.exp.str.contains('fibrin')].groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff' : d.log2fold_diff.mean(),
            'exp' : 'ECM_fibrin'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)

df_temp = pd.DataFrame()
for sg, d in df[df.exp.str.contains('collagen')].groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff' : d.log2fold_diff.mean(),
            'exp' : 'ECM_collagen'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)


#######################
# migration, amoeboid 3D
df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_sgRNA_all_2.csv')
df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
df_temp = pd.DataFrame()
for sg, d in df.groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff' : d.log2fold_diff.mean(),
            'exp' : 'ECM_fibrin_collagengood'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)

# save to disk
df_collate2.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data_ECM_2.csv')
