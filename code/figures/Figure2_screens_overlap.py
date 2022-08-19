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
# plt.style.use('styleNB')
plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']
color = ['#738FC1', '#7AA974', '#CC462F', '#EAC264', '#97459B',
         '#7CD6C4', '#D87E6A', '#BCDB8A', '#BF78C4', '#9653C1']

color_set_2 = sns.color_palette("husl", 10)
color_set_3 = sns.color_palette("Set2", 8)
# color = color_set_2

# lamtor related genes

LAMTOR_genes = ['LAMTOR2', 'LAMTOR1', 'LAMTOR3', 'LAMTOR5', 'RRAGA',
                'LAMTOR4', 'FLCN', 'RRAGD', 'RRAGC', 'RRAGA', 'RRAGB']

mTORC1_genes = ['AKT1S1', 'RPTOR', 'MTOR', 'MLST8', 'DEPTOR' ]

mTORC2_genes =  ['DEPTOR', 'MLST8', 'RICTOR', 'MAPKAP1']
gator1_genes = ['DEPDC5']# mTORC2_genes =  ['DEPTOR']

sumo_genes = ['SUMO2', 'SENP1']#, 'SENP5']

ATP_genes = ['ATP6V1C1', 'ATP6V1G1', 'ATP6V1B1', 'ATP6V1F', 'ATP6V1E1', 'ATP6V0D2', 'ATP6V0C', 'ATP6V0E1', 'ATP6V1E2', 'ATP6V1G3', 'ATP6V1B2', 'ATP6V1H', 'ATP6V1C2', 'ATP6V0B', 'ATP6V1A', 'ATP6V0E2', 'ATP6V0D1', 'ATP6V1G2']

mito_genes = ['MRPL33', 'MRPL37', 'MRPS17', 'MRPS9', 'MRPS12', 'MRPL36', 'MRPL11', 'RPL57',
'MRPS33', 'MRPL3', 'DAP3', 'MRPS14', 'MRPL19', 'MRPL13', 'MRPS5', 'MRPS2'
'MRPL14', 'MRPL16', 'MRPL35', 'MRPL32', 'MTIF2', 'MRPS21', 'MRPL41', 'MRPS7',
'MRPL43', 'MRPS15', 'MRPL55', 'MRPL46', 'MRPL10', 'MRPL17', 'MRPL34', 'MRPL15'
'MRPS23', 'MRPL50', 'MRPL24', 'MRPL38', 'MRPL9', 'MRPL53', 'MRPL45', 'MRPS11',
'MRPL30', 'MRPL21', 'MRPS24', 'MRPS31', 'MRPL42', 'MRPL22', 'MRPS28', 'MRPL49',
'MRPL27', 'MRPS26', 'MRPS18C', 'MRPL28', 'MRPS10', 'MRPL51', 'MRPL39', 'MTFMT', 'MRPL58',
'GADD45GIP1', 'MRPS35', 'MRPS27', 'MRPS22', 'MRPS34', 'MRPL12', 'PTCD3', 'MRPL48', 'MRPL54',
'MRPL23', 'MRPS18A', 'CHCHD1', 'MRPL44', 'MRPS16', 'MRPL2', 'OXA1L', 'MRPL1', 'MRPL52', 'MRPL4',
'MRPL47', 'ERAL1', 'MRPL18', 'AURKAIP1', 'MRPS30', 'MRPL40', 'MTIF3', 'MRPS25', 'MRPL20',
'MRPS18B', 'MRPS36']

electron_chain_genes = ['NDUFB9', 'BCS1L', 'NDUFA3', 'NDUFB4', 'NDUFS8', 'NDUFS1', 'NDUFA8', 'NDUFAF4', 'PET117',
'NDUFC1', 'NDUFS2', 'NDUFB6', 'NDUFB2', 'NDUFAF5', 'NDUFB5', 'NDUFA5', 'COA5', 'NDUFAF6',
'ACAD9', 'NDUFA9', 'NDUFS5', 'NDUFB10', 'NDUFS7', 'NUBPL', 'NDUFAF1', 'COX18', 'NDUFA1',
'COX20', 'NDUFAF3', 'NDUFB8', 'CHCHD4', 'TMEM126B', 'NDUFA13', 'SDHAF1', 'COX17', 'AIFM1'
'NDUFB11', 'SCO1', 'NDUFA10', 'NDUFAF8', 'SCO2', 'NDUFA6', 'NDUFS4', 'NDUFB1', 'NDUFB7',
'SDHAF4', 'COX16', 'NDUFAF2', 'TMEM126A', 'OXA1L', 'SURF1', 'COX19', 'NDUFS3', 'UQCC1'
'NDUFAF7', 'FASTKD3', 'NDUFC2', 'COA3', 'SDHAF3', 'SLC25A33', 'NDUFA2', 'NDUFA11',
'IMMP2L', 'LYRM7', 'UQCC3', 'PET100', 'UQCC2', 'NDUFAB1', 'SDHAF2', 'DUFB3', 'SMIM20',
'UQCRFS1', 'COX14', 'TAZ', 'SAMM50', 'COA1', 'TTC19', 'FOXRED1', 'TIMM21']

TF_genes = ['IRF8', 'GATA1', 'GATA2', 'RUNX1', 'GFI1', 'CEBPE', 'CEBPA', 'CEBPD', 'SPI1']


#############################################
#  Figure 2: Summary of screen results
# (A) Pathway enrichment analysis (other .py file)
# (B) Show effect size and comparison across the  screens - the  2D plots of proliferation vs
# diff, and migration vs diff.
# (E, F)  Look at cell density and survival in differentiation hits (other .py file)
#############################################


fig = plt.figure(figsize=(8.5,4))

gs = GridSpec(nrows=2, ncols=3 , height_ratios=[0.17, 1], width_ratios=[1, 1, 0.17])

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[0,1])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

###############################
# B - scatter summary across screens - effect size comparisons
###############################

#############################

# df = pd.read_csv('../../data/screen_summary/collated_screen_data_gene_pvalues_20210830.csv')
df_growth = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_growth_gene_pvalues.csv')
df_diff = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220412_screen_log2fold_diffs_differentiation_gene_pvalues.csv')

df_compare = pd.merge(df_growth,
                      df_diff,
                      on = 'gene')

sns.distplot(df_compare['log2fold_diff_mean_x'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax1)


sns.distplot(df_compare['log2fold_diff_mean_y'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax5, vertical=True)


ax1.set_xticks([])
ax1.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])

######################
# scatter - growth

ax2.spines['left'].set_position(('data', 0))
ax2.spines['bottom'].set_position(('data', 0))

ax2.scatter(df_compare.log2fold_diff_mean_x, df_compare.log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 5, label = None,
           color = '#738FC1', s = 20, alpha = 0.5)

ax2.scatter(df_compare[df_compare.gene.isin(LAMTOR_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(LAMTOR_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mTORC1 signaling',
           color = color[3], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(mito_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(mito_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mitochondrial\ntranslation',
           color = color[2], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(electron_chain_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(electron_chain_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'electron transport chain',
           color = color[1], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(sumo_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(sumo_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'sumoylation (SUMO2, SENP1)',
           color = color[6], s = 20, alpha = 1)


ax2.scatter(df_compare[df_compare.gene.isin(['MTOR'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['MTOR'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mTOR',
           color = color_set_2[7], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(['RARA'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['RARA'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'RARA',
           color = color[4], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(['CEBPA'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['CEBPA'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'CEBPA',
           color = color_set_2[5], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(['CEBPE'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['CEBPE'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'CEBPE',
           color = color_set_3[5], s = 20, alpha = 1)

ax2.scatter(df_compare[df_compare.gene.isin(['SPI1'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['SPI1'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'SPI1',
           color = color_set_3[1], s = 20, alpha = 1)

# color_set_2 = sns.color_palette("husl", 9)
# for i, gene in enumerate(['CEBPA', 'CEBPE', 'SPI1']):
#         ax2.scatter(df_compare[df_compare.gene == gene].log2fold_diff_mean_x,
#                     df_compare[df_compare.gene == gene].log2fold_diff_mean_y,
#                   edgecolors = 'k', linewidths = 0.4, zorder = 20, label = gene,
#                    color = color_set_2[i], s = 20, alpha = 1)

# ax2.scatter(df_compare[df_compare.gene.isin(TF_genes)].log2fold_diff_mean_x,
#             df_compare[df_compare.gene.isin(TF_genes)].log2fold_diff_mean_y,
#           edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'TFs (CEBPA, CEBPE, SPI1\nGFI1, GATA2)',
#            color = color[5], s = 20, alpha = 1)




# Calculate controls
d_ctrl = df_compare[df_compare.gene.str.contains('CONTROL')].copy().reset_index()

ax2.scatter(d_ctrl.log2fold_diff_mean_x, d_ctrl.log2fold_diff_mean_y,
           edgecolors = 'k', linewidths = 0.6, alpha = 1, zorder=10,
            color = '#B8BABC', label = 'controls', s = 20)




######################
# scatter - cell migration

df_mig = pd.read_csv('../../data/screen_summary/stats/gene_avg/20211222_screen_log2fold_diffs_migration_combined_gene_pvalues.csv')

df_compare = pd.merge(df_mig,
                      df_diff,
                      on = 'gene')

df_compare = df_compare[['gene', 'log2fold_diff_mean_x', 'exp_x', 'log2fold_diff_mean_y']]
df_compare.columns = ['gene', 'log2fold_diff_mean_x', 'exp', 'log2fold_diff_mean_y']

sns.distplot(df_compare['log2fold_diff_mean_x'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'legend' : False},
                  ax = ax3)
ax3.set_xticks([])
ax3.set_yticks([])



ax4.spines['left'].set_position(('data', 0))
ax4.spines['bottom'].set_position(('data', 0))

ax4.scatter(df_compare.log2fold_diff_mean_x, df_compare.log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 5,
           color = '#738FC1', s = 20, alpha = 0.5)

ax4.scatter(df_compare[df_compare.gene.isin(LAMTOR_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(LAMTOR_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mTORC1 signaling',
           color = color[3], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(mito_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(mito_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mitochondrial\ntranslation',
           color = color[2], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(electron_chain_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(electron_chain_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'electron transport chain',
           color = color[1], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(sumo_genes)].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(sumo_genes)].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'sumoylation (SUMO2, SENP1)',
           color = color[6], s = 20, alpha = 1)


ax4.scatter(df_compare[df_compare.gene.isin(['MTOR'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['MTOR'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'mTOR',
           color = color_set_2[7], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(['RARA'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['RARA'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'RARA',
           color = color[4], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(['CEBPA'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['CEBPA'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'CEBPA',
           color = color_set_2[5], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(['CEBPE'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['CEBPE'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'CEBPE',
           color = color_set_3[5], s = 20, alpha = 1)

ax4.scatter(df_compare[df_compare.gene.isin(['SPI1'])].log2fold_diff_mean_x,
            df_compare[df_compare.gene.isin(['SPI1'])].log2fold_diff_mean_y,
          edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'SPI1',
           color = color_set_3[1], s = 20, alpha = 1)

# for i, gene in enumerate(['CEBPA', 'CEBPE', 'SPI1']):
#         ax4.scatter(df_compare[df_compare.gene == gene].log2fold_diff_mean_x,
#                     df_compare[df_compare.gene == gene].log2fold_diff_mean_y,
#                   edgecolors = 'k', linewidths = 0.4, zorder = 20, label = gene,
#                    color = color_set_2[i], s = 20, alpha = 1)
#
# ax4.scatter(df_compare[df_compare.gene.isin(TF_genes)].log2fold_diff_mean_x,
#             df_compare[df_compare.gene.isin(TF_genes)].log2fold_diff_mean_y,
#           edgecolors = 'k', linewidths = 0.4, zorder = 10, label = 'TFs',
#            color = color[5], s = 20, alpha = 1)

# # Calculate controls
d_ctrl = df_compare[df_compare.gene.str.contains('CONTROL')].copy().reset_index()

ax4.scatter(d_ctrl.log2fold_diff_mean_x, d_ctrl.log2fold_diff_mean_y,
           edgecolors = 'k', linewidths = 0.6,  zorder=20,
           color = '#B8BABC', label = 'controls',
           s = 20, alpha = 1)


for ax_ in [ax1, ax2, ax3, ax4, ax5]:
    ax_.set_xlabel(None)
    ax_.set_ylabel(None)

    for axis in ['bottom','left']:
        ax_.spines[axis].set_linewidth(0.6)
    ax_.tick_params(width=0.6)

plt.tight_layout()
fig.savefig('../../figures/Fig2B_screens_scatter.pdf')

# ax2.legend()
# fig.savefig('../../figures/Fig2B_screens_scatter_wLegend.pdf')
