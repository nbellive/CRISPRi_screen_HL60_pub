import numpy as np
import pandas as pd
import random
import glob


fdir = '../../../data/screen_summary/log2foldchange/temp/'

exps = ['20220412_screen_log2fold_diffs_growth_all.csv',
        '20220412_screen_log2fold_diffs_differentiation_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_0_2hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_0_6hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_10_2hr_all.csv',
        '20220412_screen_log2fold_diffs_tracketch_10_6hr_all.csv',
        '20220516_screen_log2fold_diffs_ECM_all.csv']

df_collate = pd.DataFrame()

######################
# growth screen - just to get sgRNA dictionary and ensure it is consistent with other collate file
df = pd.read_csv(fdir + exps[0])
df = df[['gene', 'sgRNA', 'diff']]

df_temp = pd.DataFrame()
for sg, d in df.groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff_mean' : d['diff'].mean(),
            'exp' : 'growth'}
    df_temp = df_temp.append(data, ignore_index = True)

# assign unique number to each control sgRNA
sgRNA_dict = dict(zip(df[df.gene == 'CONTROL'].sgRNA.unique(),
                      np.arange(len(df[df.gene == 'CONTROL']))))

#######################
# migration, tracketch_0_all
df = pd.read_csv(fdir + exps[2])
df = df.append(pd.read_csv(fdir + exps[3]), ignore_index = True)

df = df[['gene', 'sgRNA', 'diff', 'exp']]

df_temp = pd.DataFrame()
for sg, d in df.groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff_mean' : d['diff'].mean(),
            'exp' : 'tracketch_0_all'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff_mean', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)

# save to disk
df_collate.to_csv('../../../data/screen_summary/log2foldchange/temp/collated_screen_data_misc.csv')

#######################
# migration, tracketch_10_all
df = pd.read_csv(fdir + exps[4])
df = df.append(pd.read_csv(fdir + exps[5]), ignore_index = True)
df = df[['gene', 'sgRNA', 'diff', 'exp']]

df_temp = pd.DataFrame()
for sg, d in df.groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff_mean' : d['diff'].mean(),
            'exp' : 'tracketch_10_all'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff_mean', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)

# save to disk
df_collate.to_csv('../../../data/screen_summary/log2foldchange/temp/collated_screen_data_misc.csv')

#######################
# migration, all
df = pd.read_csv(fdir + exps[2])
df = df.append(pd.read_csv(fdir + exps[3]), ignore_index = True)
df = df.append(pd.read_csv(fdir + exps[4]), ignore_index = True)
df = df.append(pd.read_csv(fdir + exps[5]), ignore_index = True)

df = df[['gene', 'sgRNA', 'diff', 'exp']]

df_temp = pd.DataFrame()
for sg, d in df.groupby(['sgRNA', 'gene']):
    data = {'gene' : sg[1],
            'sgRNA' : sg[0],
            'log2fold_diff_mean' : d['diff'].mean(),
            'exp' : 'tracketch_all'}
    df_temp = df_temp.append(data, ignore_index = True)
df_ctrl = df_temp[df_temp.gene == 'CONTROL']
df_temp = df_temp[df_temp.gene != 'CONTROL']
df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff_mean', 'exp']]
df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
df_temp = df_temp.append(df_ctrl, ignore_index = True)
df_collate = df_collate.append(df_temp, ignore_index = True)

# save to disk
df_collate.to_csv('../../../data/screen_summary/log2foldchange/temp/collated_screen_data_misc.csv')


# #######################
# # migration, all
# df = pd.read_csv(fdir + exps[2])
# df = df.append(pd.read_csv(fdir + exps[3]), ignore_index = True)
# df = df.append(pd.read_csv(fdir + exps[4]), ignore_index = True)
# df = df.append(pd.read_csv(fdir + exps[5]), ignore_index = True)
# df = df.append(pd.read_csv(fdir + exps[6]), ignore_index = True)
#
# df = df[['gene', 'sgRNA', 'diff', 'exp']]
#
# df_temp = pd.DataFrame()
# for sg, d in df.groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff_mean' : d['diff'].mean(),
#             'exp' : 'migration_all'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff_mean', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate = df_collate.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate.to_csv('../../../data/screen_summary/log2foldchange/temp/collated_screen_data_misc.csv')






































#######################
#######################
#######################
#######################
#
# df_collate2 = pd.DataFrame()
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_2.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
#
# # assign unique number to each control sgRNA
# sgRNA_dict = dict(zip(df[df.gene == 'CONTROL'].sgRNA.unique(),
#                       np.arange(len(df[df.gene == 'CONTROL']))))
#
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
# df_collate2 = df_collate2.append(df_temp, ignore_index = True)
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
# df_collate2 = df_collate2.append(df_temp, ignore_index = True)
#
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_2.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df.groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'ECM_fibrin_collagengood'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate2 = df_collate2.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate2.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data_ECM_2.csv')
#
#
# df_collate3 = pd.DataFrame()
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_3.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
#
# # assign unique number to each control sgRNA
# sgRNA_dict = dict(zip(df[df.gene == 'CONTROL'].sgRNA.unique(),
#                       np.arange(len(df[df.gene == 'CONTROL']))))
#
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
# df_collate3 = df_collate3.append(df_temp, ignore_index = True)
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
# df_collate3 = df_collate3.append(df_temp, ignore_index = True)
#
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_3.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df.groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'ECM_fibrin_collagengood'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate3 = df_collate3.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate3.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data_ECM_3.csv')

#
# df_collate4 = pd.DataFrame()
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_4.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
#
# # assign unique number to each control sgRNA
# sgRNA_dict = dict(zip(df[df.gene == 'CONTROL'].sgRNA.unique(),
#                       np.arange(len(df[df.gene == 'CONTROL']))))
#
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
# df_collate4 = df_collate4.append(df_temp, ignore_index = True)
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
# df_collate4 = df_collate4.append(df_temp, ignore_index = True)
#
#
# #######################
# # migration, amoeboid 3D
# df = pd.read_csv(fdir + '20220516_screen_log2fold_diffs_ECM_all_4.csv')
# df = df[['annotated_gene_symbol', 'sgRNA', 'diff', 'exp']]
# df.columns = ['gene', 'sgRNA', 'log2fold_diff', 'exp']
# df_temp = pd.DataFrame()
# for sg, d in df.groupby(['sgRNA', 'gene']):
#     data = {'gene' : sg[1],
#             'sgRNA' : sg[0],
#             'log2fold_diff' : d.log2fold_diff.mean(),
#             'exp' : 'ECM_fibrin_collagengood'}
#     df_temp = df_temp.append(data, ignore_index = True)
# df_ctrl = df_temp[df_temp.gene == 'CONTROL']
# df_temp = df_temp[df_temp.gene != 'CONTROL']
# df_ctrl = df_ctrl[['sgRNA', 'log2fold_diff', 'exp']]
# df_ctrl['gene'] = df_ctrl['sgRNA'].map(sgRNA_dict)
# df_ctrl['gene'] = df_ctrl.agg('CONTROL{0[gene]}'.format, axis=1)
# df_temp = df_temp.append(df_ctrl, ignore_index = True)
# df_collate4 = df_collate4.append(df_temp, ignore_index = True)
#
# # save to disk
# df_collate4.to_csv('../../../data/screen_summary/log2foldchange/collated_screen_data_ECM_4.csv')
