import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

import numpy as np
import glob
import pandas as pd
import math

import umap

import numpy as np
from sklearn.preprocessing import StandardScaler
import skimage
import skimage.io
from skimage.morphology import disk
from skimage.filters import rank

from scipy.interpolate import make_interp_spline, BSpline
from scipy import interpolate

import seaborn as sns

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']

cell_lines_marker_dict = {'sgCtrl1': 'o',
 'sgSPI1': "<",
 'sgFLCN': ">",
 'sgLAMTOR1': "p",
 'sgATIC' : "P"}

colors2 = sns.color_palette("Set2")

cell_lines_colors = {'sgControl' : '#B8BABC',
  'sgCtrl1' : '#B8BABC',
  'sgFLCN': colors2[3],
  'sgLAMTOR1':   sns.color_palette("husl", 8)[6],
  'sgSPI1' :  colors2[0],
  'sgTSC1' :  '#738FC1',
  'sgRICTOR' : '#738FC1',
  'sgCEBPE' : '#738FC1',
  'sgATIC' : '#738FC1'}

RNAseq_dict = {'WT' : 'sgCtrl1',
                'B' : 'sgFLCN',
                'C' : 'sgLAMTOR1',
                'D' : 'sgSPI1',
                'E' : 'sgATIC'}

#############################################
#  Figure S3: data on ATIC knockdown

##############################################


# df_counts_normed = pd.read_csv('../../data/rnaseq_processed/NB_counts_NBanalysis_matrix_normed_.csv')
df_counts_normed = pd.read_csv('../../data/rnaseq_processed/NB_counts_NBanalysis_matrix_normed_wATIC.csv')
df_counts_normed = df_counts_normed[df_counts_normed.columns[:-6]]
df_counts_normed = df_counts_normed.dropna()



# I'm going to transform the ATIC knockdown data into the model associated with main text (w/ data for sgControl, and LAMTOR1 & FLCN knockdown)
df_counts_normed_all = pd.read_csv('../../data/rnaseq_processed/NB_counts_NBanalysis_matrix_normed_wATIC.csv')
df_counts_normed_all['index_'] = np.arange(len(df_counts_normed_all))
df_counts_normed_all = df_counts_normed_all[np.append(df_counts_normed_all.columns[-1], df_counts_normed_all.columns[:-1])]
df_counts_normed_all = df_counts_normed_all[df_counts_normed_all.columns[:-6]]
df_counts_normed_all = df_counts_normed_all.dropna()


celltypes = ['WT', 'WT', 'WT', 'WT', 'WT', 'WT',
       'WT', 'WT', 'WT', 'WT', 'WT', 'WT',
       'WT', 'WT', 'WT', 'WT', 'WT', 'WT',
       'WT', 'WT', 'WT', 'WT', 'WT', 'WT',
       'B', 'B', 'B', 'B', 'B', 'B', 'B',
       'B', 'B', 'B', 'B', 'B', 'B', 'B',
       'B', 'B', 'B', 'B', 'B', 'B', 'B',
       'B', 'B', 'B',
       'C', 'C', 'C', 'C',
       'C', 'C', 'C', 'C', 'C', 'C', 'C',
       'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
       'C', 'C', 'C', 'C', 'D', 'D', 'D',
       'D', 'D', 'D', 'D', 'D', 'D', 'D',
       'D', 'D', 'D', 'D', 'D', 'D', 'D',
       'D', 'C']



# Try iterating over different seed values
fig, ax = plt.subplots(figsize=(5,5))

reducer = umap.UMAP(n_components = 2, random_state=42)

# normalization; I think this approach makes most ense since we already normed count data using deseq2
RNA_data = df_counts_normed.dropna().values.T

# perform dimension reduction on z-scores
scaled_RNA_data = StandardScaler().fit_transform(RNA_data)

embedding = reducer.fit_transform(scaled_RNA_data)

color_dict = {'WT': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

map_df = pd.DataFrame(embedding, columns=('1', '2'))
map_df['celltype'] = celltypes
map_df['condition'] = df_counts_normed.columns

# the UMAP axis are arbitrary  units; UMAP1 shows ~time dimension - make it go from left to  right
map_df['1'] = -1*map_df['1']

# create spline lines to connect pseudo-time
for c, d in map_df.groupby('celltype', sort = False):
    col = cell_lines_colors[RNAseq_dict[c]]


    d0 = d[d.condition.str.contains('d0')]
    print(d0['1'].mean(), d0['1'].std())
    d0_UMAP1 = d0['1'].mean()
    d0_UMAP2 = d0['2'].mean()
    d0_UMAP1_err = d0['1'].std()
    d0_UMAP2_err = d0['2'].std()

    d1 = d[d.condition.str.contains('d1')]
    d1_UMAP1 = d1['1'].mean()
    d1_UMAP2 = d1['2'].mean()
    d1_UMAP1_err = d1['1'].std()
    d1_UMAP2_err = d1['2'].std()

    d5 = d[d.condition.str.contains('d5')]
    d5_UMAP1 = d5['1'].mean()
    d5_UMAP2 = d5['2'].mean()
    d5_UMAP1_err = d5['1'].std()
    d5_UMAP2_err = d5['2'].std()

    if c != 'D':
        d7 = d[d.condition.str.contains('d7')]
        d7_UMAP1 = d7['1'].mean()
        d7_UMAP2 = d7['2'].mean()
        d7_UMAP1_err = d7['1'].std()
        d7_UMAP2_err = d7['2'].std()

        UMAP1_avg = np.array([d0_UMAP1, d1_UMAP1, d5_UMAP1, d7_UMAP1])
        UMAP2_avg = np.array([d0_UMAP2, d1_UMAP2, d5_UMAP2, d7_UMAP2])
        UMAP1_std = np.array([d0_UMAP1_err, d1_UMAP1_err, d5_UMAP1_err, d7_UMAP1_err])
        UMAP2_std = np.array([d0_UMAP2_err, d1_UMAP2_err, d5_UMAP2_err, d7_UMAP2_err])

    else:
        UMAP1_avg = np.array([d0_UMAP1, d1_UMAP1, d5_UMAP1])
        UMAP2_avg = np.array([d0_UMAP2, d1_UMAP2, d5_UMAP2])
        UMAP1_std = np.array([d0_UMAP1_err, d1_UMAP1_err, d5_UMAP1_err])
        UMAP2_std = np.array([d0_UMAP2_err, d1_UMAP2_err, d5_UMAP2_err])

    #   plot avg datapoints
    ax.errorbar(UMAP1_avg, UMAP2_avg, xerr = UMAP1_std, yerr = UMAP2_std,
    color = col, marker = cell_lines_marker_dict[RNAseq_dict[c]], markeredgecolor = 'k',
    markeredgewidth = 1, label = RNAseq_dict[c], linewidth = 0, elinewidth = 1,zorder=10,
    alpha = 1.0, markersize = 10)


    x = UMAP1_avg
    y = UMAP2_avg

    tck,u     = interpolate.splprep([x, y], k = 2)
    xnew, ynew = interpolate.splev(np.linspace(x.min(), x.max(), num=500), tck)

    df_temp = pd.DataFrame()
    df_temp['x'] = xnew
    df_temp['y'] = ynew
    df_temp = df_temp[df_temp['x'] >= x.min()]
    df_temp = df_temp[df_temp['x'] <= x.max()]
    df_temp = df_temp[df_temp['y'] >= y.min()]
    if c == 'D':
        df_temp = df_temp[df_temp['y'] <= y.max()]
    ax.plot(df_temp.x.values, df_temp.y.values,
            color = col,
            zorder=0, linestyle = '--')#, alpha = 0.5)


legend_without_duplicate_labels(ax)
ax.set_xlabel('UMAP mode 1')
ax.set_ylabel('UMAP mode 2')

ax.set_aspect('equal')


#######################
#  Add in ATIC data point
#######################

df_counts_normed_ATIC = pd.read_csv('../../data/rnaseq_processed/NB_counts_NBanalysis_matrix_normed_wATIC.csv')
df_counts_normed_ATIC['index_'] = np.arange(len(df_counts_normed_ATIC))

df_counts_normed_ATIC = df_counts_normed_ATIC[['index_','WT_d0_1', 'E_d0_1','E_d0_2','E_d0_3','E_d0_4','E_d0_5','E_d0_6']]
df_counts_normed_ATIC = df_counts_normed_ATIC[df_counts_normed_ATIC.index_.isin(df_counts_normed_all.index_.values)]
df_counts_normed_ATIC = df_counts_normed_ATIC.fillna(0)

df_counts_normed_ATIC = df_counts_normed_ATIC[['E_d0_1','E_d0_2','E_d0_3','E_d0_4','E_d0_5','E_d0_6']]


celltypes_ATIC = ['E', 'E', 'E', 'E', 'E', 'E']

# normalization; I think this approach makes most sense since we already normed count data using deseq2
RNA_data_ATIC = df_counts_normed_ATIC.dropna().values.T

# perform dimension reduction on z-scores
embedding_ATIC = reducer.transform(RNA_data_ATIC)

color_dict = {'E': 4}

map_df = pd.DataFrame(embedding_ATIC, columns=('1', '2'))
map_df['celltype'] = celltypes_ATIC
map_df['condition'] = df_counts_normed_ATIC.columns

# the UMAP axis are arbitrary  units; UMAP1 shows ~time dimension - make it go from left to  right
map_df['1'] = -1*map_df['1']

# create spline lines to connect pseudo-time
for c, d in map_df.groupby('celltype', sort = False):
    col = cell_lines_colors[RNAseq_dict[c]]

    d0 = d[d.condition.str.contains('d0')]
    d0_UMAP1 = d0['1'].mean()
    d0_UMAP2 = d0['2'].mean()
    d0_UMAP1_err = d0['1'].std()
    d0_UMAP2_err = d0['2'].std()

    #   plot avg datapoints
    ax.errorbar(d0_UMAP1, d0_UMAP2, xerr = d0_UMAP1_err, yerr = d0_UMAP2_err,
    color = col, marker = cell_lines_marker_dict[RNAseq_dict[c]], markeredgecolor = 'k',
    markeredgewidth = 1, label = 'sgATIC, uHL-60', linewidth = 0, elinewidth = 1,zorder=10,
    alpha = 1.0, markersize = 10)

ax.legend()


plt.tight_layout()
fig.savefig('../../figures/FigS3_ATIC_rnaseq_test.pdf', bbox_inches='tight')
