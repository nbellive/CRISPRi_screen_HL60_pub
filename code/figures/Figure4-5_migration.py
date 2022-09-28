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
from scipy import stats

import seaborn as sns

plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
color = ['#738FC1', '#7AA974', '#CC462F', '#EAC264', '#97459B',
         '#7CD6C4', '#D87E6A', '#BCDB8A', '#BF78C4', '#9653C1']


#############################################
#  Figure 4: Summary of '2D' cell migration screen results
# (A) Show scatter plots comparing chemotaxis vs. chemokinesis
# (B) Schematic of integrin alphaM-beta2 signaling
# (C) phase images of 2D migration in sgControl1 versus sgITGB2
# (D) Comparison with individual cell lines in track-etch chemotaxis (10% FBS gradient)
# (E) uHL-60 sgATIC phase 20x
# (F) analysis of sgGIT2 cell line

# Figure 5: Summary of '3D' cell migration screen results
# (A) ECM vs chemokinesis
# (B) Comparison of screen data with analysis of individual cell lines
# (C/D) immunofluorescence of FMNL1 and CORO1A
#############################################


fig1 = plt.figure(figsize=(6.5,3))
gs1 = GridSpec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[0.6,0.7,0.7]) # for (b)

ax_A   = fig1.add_subplot(gs1[0])
ax_Bi  = fig1.add_subplot(gs1[1])
ax_Bii = fig1.add_subplot(gs1[2])

###############################
# A -  of several sgRNA in track-etch stype screen (chemotaxis)
###############################
# some useful dictionaries for sgRNA and markertype

cell_lines_sg = ['AGGGCACCCGGTTCATACGC',
'CTGGGTTCAGGGCGAGCGGG',
'CGGTGTGCTGGAGTCCTCGG',
'GGCGGCGCTTCCGCTCTAAC',
'GGGCGACCCGAGAAGCGGCG',
'CAGGACACAATTTCTTGCCA',
'CCTTAGTCCCTCTTGCGTCG',
'GCCCCGTCCGTGGGACCGGG',
'CGACGGTGGTGGTGACTGAG']

cell_lines_sg_dict = {'sgCTRL1': 'AGGGCACCCGGTTCATACGC',
 'sgATIC': 'CTGGGTTCAGGGCGAGCGGG',
 'sgITGB2': 'CGGTGTGCTGGAGTCCTCGG',
 'sgGIT2': 'GGCGGCGCTTCCGCTCTAAC',
 'sgTLN1': 'GGGCGACCCGAGAAGCGGCG',
 'sgARHGAP30': 'CAGGACACAATTTCTTGCCA',
 'sgAPBB1IP': 'CCTTAGTCCCTCTTGCGTCG',
 'sgFMNL1': 'GCCCCGTCCGTGGGACCGGG',
 'sgVPS29': 'CGACGGTGGTGGTGACTGAG'}

cell_lines_marker_dict = {'sgCTRL1': 'o',
 'sgATIC':  "*",
 'sgITGB2': "v",
 'sgGIT2': 'X',
 'sgTLN1': 'd',
 'sgARHGAP30': 'H',
 'sgAPBB1IP':  "<",
 'sgFMNL1': "^",
 'sgVPS29': ">"}

pearson_screen = np.zeros(9)
pearson_transwell = np.zeros(9)
pearson_gene_dict = {'sgCTRL1': 7,
 'sgATIC':  5,
 'sgITGB2': 0,
 'sgGIT2': 8,
 'sgTLN1': 2,
 'sgARHGAP30': 4,
 'sgAPBB1IP':  1,
 'sgFMNL1': 6,
 'sgVPS29': 3}

# grab experimental data from individual transwell assays for each knockdown line
df = pd.read_csv('../../data/screen_followups/20211122_KDlines_transwell.csv')
df = df[df.percent_migrated <= 40]

# screen  data - grab only 10% gradient
# df_screen = pd.read_csv('../../data/screen_summary/collated_screen_data_sgRNA_20211222.csv')
df_screen = pd.read_csv('../../data/screen_summary/log2foldchange/collated_screen_data_sgRNA.csv')
df_screen =  df_screen.drop_duplicates()
df_screen =  df_screen[df_screen.exp.str.contains('tracketch')]
df_screen =  df_screen[df_screen.exp.str.contains('_10_')]

# only worry about sgRNA from individual lines tested
df_screen = df_screen[df_screen.sgRNA.isin(cell_lines_sg)]

# plot  data
for gene, d in df.groupby('cell_line'):
    if gene == 'sgCTRL1':
        N_screen = len(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff)
        N_tracketch = len(d.percent_migrated.values)

        ax_A.errorbar(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.mean(),
                    d.percent_migrated.mean()/100,
                    xerr = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.std()/N_screen**0.5,
                    yerr = (d.percent_migrated.std()/100)/N_tracketch**0.5,
                   markersize = 7, marker = cell_lines_marker_dict[gene], color = '#B8BABC',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1, label = gene)
        ctrl_value = d.percent_migrated.mean()/100
        pearson_screen[pearson_gene_dict[gene]] = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.mean()
        pearson_transwell[pearson_gene_dict[gene]] = d.percent_migrated.mean()/100
    else:
        continue

for gene, d in df.sort_values(by='percent_migrated').groupby('cell_line', sort=False):
    if gene == 'sgCTRL1':
        continue
    else:
        N_screen = len(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.values)
        N_tracketch = len(d.percent_migrated.values)
        ax_A.errorbar(df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.mean(),
                   d.percent_migrated.mean()/100,
                    xerr = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.std()/N_screen**0.5,
                    yerr = (d.percent_migrated.std()/100)/N_tracketch**0.5,
                    marker =  cell_lines_marker_dict[gene],
                    color  = '#7AA974', markersize = 7, markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1, label = gene)
        pearson_screen[pearson_gene_dict[gene]] = df_screen[df_screen.sgRNA == cell_lines_sg_dict[gene]].log2fold_diff.mean()
        pearson_transwell[pearson_gene_dict[gene]] = d.percent_migrated.mean()/100

ax_A.set_xlabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_A.set_ylabel('fraction of cells migrated\n(transwell, 10% FBS gradient)')

ax_A.set_ylim(0,0.40)
# ax.set_xlim(-3.25,1.5)
# ax_A.set_xticks([-5,-4,-3,-2,-1,0,1,2,3])
ax_A.set_xticks([-4,-2,0,2])
ax_A.axhline(ctrl_value,xmin =-3.25, xmax = 1.5, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )
ax_A.axvline(0,ymin = 0, ymax = 1, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )

# ax_A.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')#, fontsize = 14)

print('Pearson\ncorrelation\n' + r'$\rho$ = '
           + str(np.round(stats.pearsonr(pearson_transwell,
                                         pearson_screen), 5)))
print(pearson_transwell, pearson_screen)

###############################
# B - validation of several sgRNA in ECM screen
###############################

# some useful dictionaries for sgRNA and markertype

cell_lines_sg = ['AGGGCACCCGGTTCATACGC',
'CGGTGTGCTGGAGTCCTCGG',
'GCCCCGTCCGTGGGACCGGG',
'GCCCGGGTTCAGGCTCTCAG',
'GCTGCTGTAGCAGCACCCCA',
'ATCTTCAGCGGGCGAGTCCC',
'CGTGTTTAGGCTAAAGTCCA']

cell_lines_sg_dict = {'sgControl1': 'AGGGCACCCGGTTCATACGC',
 'sgITGB2': 'CGGTGTGCTGGAGTCCTCGG',
 'sgFMNL1': 'GCCCCGTCCGTGGGACCGGG',
 'sgFLCN' : 'GCCCGGGTTCAGGCTCTCAG',
 'sgLAMTOR1' : 'GCTGCTGTAGCAGCACCCCA',
 'sgCORO1A' : 'ATCTTCAGCGGGCGAGTCCC',
 'sgITGA1' : 'CGTGTTTAGGCTAAAGTCCA'}

cell_lines_marker_dict = {'sgControl1': 'o',
 'sgITGB2': "v",
 'sgFMNL1': "^",
 'sgFLCN' : "*",
 'sgLAMTOR1' : 'X',
 'sgCORO1A' : 'd',
 'sgITGA1' : 'H'}

pearson_screen_3D = np.zeros(4)
pearson_ecm_speed = np.zeros(4)
pearson_ecm_persistence = np.zeros(4)
pearson_gene_dict_3D = {'Control1': 3,
 'CORO1A':  1,
 'FMNL1': 0,
 'ITGB2' : 2}


# we will be comparing the screen data to average migration speed in 3D.
# I've tabulated the average speed of cells tracked during 3D migration and
# can load that data.
files_bay_3d = glob.glob('../../data/processed_tracking_bayesian/2022*_3D_filtered_xyzCorr_avg*all*')

df_3D = pd.DataFrame()

for f in files_bay_3d:
    if 'AG++' in f:
        continue
    df_temp = pd.read_csv(f)
    df_3D = df_3D.append(df_temp, ignore_index = True)

# standard collagen concentration tested across all cell lines is 0.75 mg/ml collagen
df_3D =  df_3D[df_3D.concentration == '0.75mgml']

# Load in ECM screen data - here is the mean log fold values
# df_screen3D = pd.read_csv('../../data/screen_summary/final/20220516_screen_log2fold_diffs_ECM_truncate_sgRNA_means.csv')
# df_screen3D = df_screen3D.drop_duplicates()
df_screen3D = pd.read_csv('../../data/screen_summary/temp/collated_screen_data_ECM_3.csv')
df_screen3D = df_screen3D[df_screen3D.sgRNA.isin(cell_lines_sg)]

# Load in ECM screen data - here is the individual log fold values across all experiments
# df_screen3D_err = pd.read_csv('../../data/screen_summary/final/20220516_screen_log2fold_diffs_ECM_truncate_sgRNA_all.csv')
# df_screen3D_err = pd.read_csv('../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_sgRNA_all.csv')
# df_screen3D_err = pd.read_csv('../../data/screen_summary/temp/collated_screen_data_ECM_3.csv')
df_screen3D_err = pd.read_csv('../../data/screen_summary/log2foldchange/20220516_screen_log2fold_diffs_ECM_sgRNA_all.csv')
df_screen3D_err =  df_screen3D_err.drop_duplicates()
# df_screen3D_err =  df_screen3D_err[df_screen3D_err.exp.str.contains('ECM')]
# print(df_screen3D_err)

# plot  data
for c in ['Control1', 'CORO1A', 'FMNL1',  'ITGB2']:
    d_3D = df_3D[df_3D.celltype.str.contains(c)]
    if c  ==  'Control1':
        col = '#B8BABC'
        d_screen3 = df_screen3D[df_screen3D.sgRNA == 'AGGGCACCCGGTTCATACGC']
        d_screen3_err = df_screen3D_err[df_screen3D_err.sgRNA == 'AGGGCACCCGGTTCATACGC']
    elif c == 'ITGB2':
        col = '#7AA974'
        d_screen3 = df_screen3D[df_screen3D.gene == c]
        d_screen3_err = df_screen3D_err[df_screen3D_err.annotated_gene_symbol == c]
        d_screen3_err  = d_screen3_err[d_screen3_err.sgRNA  == cell_lines_sg_dict['sg'+c]]
    else:
        col =  '#EAC264' #'#738FC1'
        d_screen3 = df_screen3D[df_screen3D.gene == c]
        d_screen3_err = df_screen3D_err[df_screen3D_err.annotated_gene_symbol == c]
        d_screen3_err  = d_screen3_err[d_screen3_err.sgRNA  == cell_lines_sg_dict['sg'+c]]

    # calculate average speed across tracked cells
    N_3d = 0
    speeds = []
    for i, d in d_3D.groupby(['date', 'trial']):
        N_3d += 1
        speeds = np.append(speeds, np.median(d['speed ($\mu$m/sec)']))

    ax_Bi.errorbar(d_screen3.log2fold_diff.mean(),  np.median(d_3D['speed ($\mu$m/sec)'].values),
        xerr = d_screen3_err['diff'].std()/np.sqrt(len(d_screen3_err['diff'])),
        yerr = np.std(speeds)/np.sqrt(N_3d),
       markersize = 7, marker = cell_lines_marker_dict['sg'+c], color = col,
        markeredgecolor = 'k',
        markeredgewidth = 0.5, lw = 1, label = c)

    pearson_screen_3D[pearson_gene_dict_3D[c]] = d_screen3.log2fold_diff.mean()
    pearson_ecm_speed[pearson_gene_dict_3D[c]] = np.median(d_3D['speed ($\mu$m/sec)'].values)

    if c  ==  'Control1':
        ctrl_value = np.mean(speeds)


ax_Bi.set_xlabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Bi.set_ylabel('median cell speed\nin collagen ['
                r'$\mu$m/s]')

ax_Bi.set_xlim(-3.4,0.5)
ax_Bi.set_ylim(0.05,0.16)
ax_Bi.set_xticks([-3,-2,-1,0])
ax_Bi.axhline(ctrl_value,xmin =-3.25, xmax = 1.5, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )
ax_Bi.axvline(0,ymin = 0, ymax = 1, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )


print('ECM, Pearson\ncorrelation\n' + r'$\rho$ = '
           + str(np.round(stats.pearsonr(pearson_ecm_speed,
                                         pearson_screen_3D), 5)))
print(pearson_ecm_speed, pearson_screen_3D)

#############################

# plot  data
for c in ['Control1', 'CORO1A', 'FMNL1',  'ITGB2']:
    d_3D = df_3D[df_3D.celltype.str.contains(c)]
    if c  ==  'Control1':
        col = '#B8BABC'
        d_screen3 = df_screen3D[df_screen3D.sgRNA == 'AGGGCACCCGGTTCATACGC']
        d_screen3_err = df_screen3D_err[df_screen3D_err.sgRNA == 'AGGGCACCCGGTTCATACGC']
    elif c == 'ITGB2':
        col = '#7AA974'
        d_screen3 = df_screen3D[df_screen3D.gene == c]
        d_screen3_err = df_screen3D_err[df_screen3D_err.annotated_gene_symbol == c]
        d_screen3_err  = d_screen3_err[d_screen3_err.sgRNA  == cell_lines_sg_dict['sg'+c]]
    else:
        col =  '#EAC264' #'#738FC1'
        d_screen3 = df_screen3D[df_screen3D.gene == c]
        d_screen3_err = df_screen3D_err[df_screen3D_err.annotated_gene_symbol == c]
        d_screen3_err  = d_screen3_err[d_screen3_err.sgRNA  == cell_lines_sg_dict['sg'+c]]

    # calculate average speed across tracked cells
    N_3d = 0
    speeds = []
    for i, d in d_3D.groupby(['date', 'trial']):
        N_3d += 1
        speeds = np.append(speeds, np.median(d['average_persistence']))

    ax_Bii.errorbar(d_screen3.log2fold_diff.mean(), np.median(d_3D['average_persistence']),#np.mean(speeds),
        xerr = d_screen3_err['diff'].std()/np.sqrt(len(d_screen3_err['diff'])),
        yerr = np.std(speeds)/np.sqrt(N_3d),#np.std(d_3D['speed ($\mu$m/sec)'])/np.sqrt(N_3d),
       markersize = 7, marker = cell_lines_marker_dict['sg'+c], color = col,
        markeredgecolor = 'k',
        markeredgewidth = 0.5, lw = 1, label = c)

    pearson_screen_3D[pearson_gene_dict_3D[c]] = d_screen3.log2fold_diff.mean()
    pearson_ecm_persistence[pearson_gene_dict_3D[c]] = np.median(d_3D['average_persistence'].values)


    if c  ==  'Control1':
        ctrl_value = np.mean(speeds)

ax_Bii.set_xlabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Bii.set_ylabel('median cell persistence\nin collagen')

ax_Bii.set_xlim(-3.4,0.5)
ax_Bii.set_ylim(-0.1,0.4)
ax_Bii.set_xticks([-3,-2,-1,0])
ax_Bii.axhline(ctrl_value,xmin =-3.25, xmax = 1.5, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )
ax_Bii.axvline(0,ymin = 0, ymax = 1, lw = 0.5, ls = '--', alpha = 0.5, zorder = 0 )


print('ECM, Pearson\ncorrelation\n' + r'$\rho$ = '
           + str(np.round(stats.pearsonr(pearson_ecm_persistence,
                                         pearson_screen_3D), 5)))
print(pearson_ecm_persistence, pearson_screen_3D)


plt.tight_layout()
fig1.savefig('../../figures/Figure4D-5B_migration_screen_summaries.pdf')#, bbox_inches='tight')
#

# ###############################
# ###############################
# ###############################

fig2 = plt.figure(figsize=(5.75,2.8))
gs2 = GridSpec(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1,1])
ax_Ci = fig2.add_subplot(gs2[0])
ax_Cii = fig2.add_subplot(gs2[1])

###############################
# C - Comparison of chemotaxis, chemokinesis, and ECM
###############################

df_10 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20211222_screen_log2fold_diffs_tracketchcombined_10grad_gene_pvalues.csv')
df_0 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20211222_screen_log2fold_diffs_tracketchcombined_Nograd_gene_pvalues.csv')

df_compare = pd.merge(df_10,
                      df_0,
                      on = 'gene')


ax_Ci.scatter(df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_x,
            df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = None,
           color = color[0], s = 20, alpha = 0.5)


# pseudogenes
d_ctrl = df_compare[df_compare.gene.str.contains('CONTROL')].copy().reset_index()
ax_Ci.scatter(d_ctrl.log2fold_diff_mean_x, d_ctrl.log2fold_diff_mean_y,
           edgecolors = 'k', linewidths = 0.6, alpha = 1, zorder=10,
            color = '#B8BABC', label = 'control pseudogenes', s = 20)


# ax_Ci.set_xlim(-2.9,2.1)
# ax_Ci.set_ylim(-2.9,2.1)
ax_Ci.spines['left'].set_position(('data', 0))
ax_Ci.spines['bottom'].set_position(('data', 0))

# ax_Ci.set_xlabel(r'normalized log$_{2}$(fold-change)')
# ax_Ci.set_ylabel(r'normalized log$_{2}$(fold-change)')
ax_Ci.set_xlabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Ci.set_ylabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Ci.xaxis.set_label_coords(0.6, -0.05)
ax_Ci.yaxis.set_label_coords(0.0, 0.6)

# df_compare = pd.merge(df[df.exp == 'ECM_all_data'],
#                       df[df.exp == 'transwell_all_NOgrad_data'],
#                       on = 'gene')
#
# ax5.scatter(df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_x,
#             df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_y,
#           edgecolors = 'k', linewidths = 0.4, zorder = 10, label = None,
#            color = color[0], s = 20, alpha = 0.5)
#
# # pseudogenes
# d_ctrl = df_compare[df_compare.gene.str.contains('CONTROL')].copy().reset_index()
# ax5.scatter(d_ctrl.log2fold_diff_mean_x, d_ctrl.log2fold_diff_mean_y,
#            edgecolors = 'k', linewidths = 0.6, alpha = 1, zorder=10,
#             color = '#B8BABC', label = 'control pseudogenes', s = 20)

# df = df.append(pd.read_csv('../../data/screen_summary/collated_screen_data_gene_pvalues_20210830.csv'),
#                 ignore_index = True)

df_0 = pd.read_csv('../../data/screen_summary/stats/gene_avg/20211222_screen_log2fold_diffs_tracketchcombined_Nograd_gene_pvalues.csv')
df_ECM = pd.read_csv('../../data/screen_summary/temp/20211222_screen_log2fold_diffs_ECM_gene_pvalues_3.csv')

df_compare = pd.merge(df_ECM,
                      df_0,
                      on = 'gene')

ax_Cii.scatter(df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_x,
            df_compare[~df_compare.gene.str.contains('CONTROL')].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = None,
           color = color[0], s = 20, alpha = 0.5)

# controls
d_ctrl = df_compare[df_compare.gene.str.contains('CONTROL')].copy().reset_index()
ax_Cii.scatter(d_ctrl.log2fold_diff_mean_x, d_ctrl.log2fold_diff_mean_y,
           edgecolors = 'k', linewidths = 0.6, alpha = 1, zorder=10,
            color = '#B8BABC', label = 'control pseudogenes', s = 20)

# ax_Cii.set_xlim(-2.5,1.5)
# ax_Cii.set_ylim(-2.9,2.1)
ax_Cii.spines['left'].set_position(('data', 0))
ax_Cii.spines['bottom'].set_position(('data', 0))
ax_Cii.set_xlabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Cii.set_ylabel('normalized\n'
                r'log$_{2}$(fold-change)')
ax_Cii.xaxis.set_label_coords(0.6, -0.05)
ax_Cii.yaxis.set_label_coords(0.0, 0.6)

plt.tight_layout()
fig2.savefig('../../figures/Figure4A-5A_migration_screen_comparison.pdf')#, bbox_inches='tight')



###############################
# Git2 2D migration analysis
###############################
fig3 = plt.figure(figsize=(2.25,1.5))
gs3 = GridSpec(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1,1])

ax_git2_s   = fig3.add_subplot(gs3[0])
ax_git2_p   = fig3.add_subplot(gs3[1])



files_bay_2d = glob.glob('../../data/processed_tracking_bayesian/*_2D_filtered_cellAvg*all*')

df_Bayes_2d = pd.DataFrame()
for f in files_bay_2d:
    if np.any(['Control1' in f, 'GIT2' in f]):
        print(f)
        df_temp = pd.read_csv(f)
        df_Bayes_2d = df_Bayes_2d.append(df_temp, ignore_index = True)
df_Bayes_2d = df_Bayes_2d[['average_persistence', 'cell', 'celltype',
                'date', 'position', 'speed ($\mu$m/sec)', 'trial']]
df_Bayes_2d.columns = ['average_persistence', 'cell', 'celltype',
                'date', 'position', 'average_speed', 'trial']


########
# speed
########
lines_2D = ['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgGIT2']
vp_2d_s = np.empty_like(np.array([ax_git2_s,ax_git2_s]))

pos = [0.5, 1, 1.75, 2.25, 2.75]
for i, ct in enumerate(lines_2D):
    y = df_Bayes_2d[df_Bayes_2d.celltype == ct]['average_speed']
    print(i, y.mean())
    vp_2d_s[i] = ax_git2_s.violinplot(y, positions = [pos[i]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#7AA974'

    for pc in vp_2d_s[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax_git2_s.hlines(y.mean(), pos[i]-0.15,pos[i]+0.15, zorder=10, lw = 1.5)

    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        ax_git2_s.errorbar(np.random.normal(pos[i], 0.05, 1), d['average_speed'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.15)


# ax.set_xlim(0,2)
ax_git2_s.set_ylim(0.0,0.45)
# ax_git2_s.set_ylabel('average speed\n' + r'[$\mu$m/s]', fontsize = 10)
ax_git2_s.spines['bottom'].set_position(('data', 0))
ax_git2_s.set_xticks([])

########
# persistence
########
vp_2d_p = np.empty_like(np.array([ax_git2_s,ax_git2_s]))

pos = [0.5, 1, 1.75, 2.25, 2.75]
for i, ct in enumerate(lines_2D):
    y = df_Bayes_2d[df_Bayes_2d.celltype == ct]['average_persistence']
    print(i, y.mean())
    vp_2d_p[i] = ax_git2_p.violinplot(y, positions = [pos[i]], points=60, widths=0.3,
                         showmeans=False, showextrema=False, showmedians=False,
                         bw_method=0.2)
    if 'Control' in ct:
        color = '#B8BABC'
    else:
        color = '#7AA974'

    for pc in vp_2d_p[i]['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.5)

    ax_git2_p.hlines(y.mean(), pos[i]-0.15,pos[i]+0.15, zorder=10, lw = 1.5)

    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        ax_git2_p.errorbar(np.random.normal(pos[i], 0.05, 1), d['average_persistence'].mean(),
                    markersize = 4, marker = 'o', color  = 'k',
                        markeredgecolor = 'k',
                        markeredgewidth = 0.5,
                       lw = 0.5,
                      alpha = 0.15)

# ax.set_xlim(0,2)
ax_git2_p.set_ylim(-0.5,1.0)
ax_git2_p.spines['bottom'].set_position(('data', 0))
ax_git2_p.set_xticks([])

plt.tight_layout()
fig3.savefig('../../figures/Fig4F_migration_GIT2.pdf')


########################
########################
# statistics
########################
########################
import scipy

param = 'average_speed'
exp_means_2d_wt = []
exp_means_2d_GIT2 = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_GIT2 = np.append(exp_means_2d_GIT2, d[param].mean())

print('2D, speed, GIT2', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_GIT2))


param = 'average_persistence'
exp_means_2d_wt = []
exp_means_2d_b2 = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_GIT2 = np.append(exp_means_2d_GIT2, d[param].mean())

print('2D, persistence, GIT2', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_GIT2))
