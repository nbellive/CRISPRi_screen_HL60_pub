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
 'sgLAMTOR1': "p"}

colors2 = sns.color_palette("Set2")

cell_lines_colors = {'sgControl' : '#B8BABC',
  'sgCtrl1' : '#B8BABC',
  'sgFLCN': colors2[3],
  'sgLAMTOR1':   sns.color_palette("husl", 8)[6],
  'sgSPI1' :  colors2[0],
  'sgTSC1' :  '#738FC1',
  'sgRICTOR' : '#738FC1',
  'sgCEBPE' : '#738FC1'}

RNAseq_dict = {'WT' : 'sgCtrl1',
                'B' : 'sgFLCN',
                'C' : 'sgLAMTOR1',
                'D' : 'sgSPI1'}

#############################################
#  Figure 7: Understanding the changes in candidates like LAMTOR1 and FLCN
# (A) Western blot related to mTORC1
# (B) schematic of RNA-seq experiment and UMAP result
# (C) Venn diagram to show number of significantly changed genes at day 5/7?; maybe just a bar plot?
# (D) Schematic to show trend in AMPK and autophagy
# (E) Enrichment analysis
# (F) Rapamycin treatment experiments to look at lifetime

# row 1:
# row 2:
# row 3: (F)
#############################################

#
fig = plt.figure(figsize=(7,5))
gs = GridSpec(nrows=2, ncols=6, height_ratios=[1.5, 1], width_ratios=[1, 1, 1, 1, 1, 1])

ax_B   = fig.add_subplot(gs[0,-3:])

ax_Fi    = fig.add_subplot(gs[1,:2])
ax_Fiii   = fig.add_subplot(gs[1,2:4])
ax_Fii  = fig.add_subplot(gs[1,4:])


###############################
# B - UMAP
###############################
df_counts_normed = pd.read_csv('../../data/rnaseq_processed/NB_counts_NBanalysis_matrix_normed_.csv')
df_counts_normed = df_counts_normed.dropna()

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


reducer = umap.UMAP(n_components = 2, random_state=42)

# normalization; I think this approach makes most ense since we already normed count data using deseq2
RNA_data = df_counts_normed.dropna().values.T
# perform dimension reduction on z-scores
scaled_RNA_data = StandardScaler().fit_transform(RNA_data)

embedding = reducer.fit_transform(scaled_RNA_data)

color_dict = {'WT': 0, 'B': 1, 'C': 2, 'D': 3}

map_df = pd.DataFrame(embedding, columns=('1', '2'))
map_df['celltype'] = celltypes
map_df['condition'] = df_counts_normed.columns

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
    ax_B.errorbar(UMAP1_avg, UMAP2_avg, xerr = UMAP1_std, yerr = UMAP2_std,
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
    ax_B.plot(df_temp.x.values, df_temp.y.values,
            color = col,
            zorder=0, linestyle = '--')#, alpha = 0.5)
    # ax_B.set_xlim(-20,25)
    # ax_B.set_ylim(-20,30)

legend_without_duplicate_labels(ax_B)
ax_B.set_xlabel('UMAP mode 1')
ax_B.set_ylabel('UMAP mode 2')

ax_B.set_aspect('equal')
#############################
#############################
# () Cell lifetime experiments
#############################
#############################


files = glob.glob('../../data/screen_followups/20211122_proliferation_curves.csv')
df_growth = pd.read_csv(files[0])
df_growth.head()

df_growth = df_growth[df_growth.cell_line != 'sgSPI1']
df_growth = df_growth[df_growth.cell_line != 'sgCEBPE']
df_growth.date.unique()
day_dict = {'20111116' : 0,
            '20111117': 1,
            '20111118': 2,
            '20111119': 3,
            '20111120': 4,
            '20111121': 5,
            '20111123': 7,
            '20111126': 10,
            '20211129': 13}

df_growth['trial'] = 1
df_growth['date_'] = df_growth['date'].astype('str')
df_growth['day_diff'] = df_growth['date_'].map(day_dict)

df_ = df_growth[df_growth.DMSO == True]
df_['rapa_nM'] = 0.0
df_ = df_[['effective_density_1E6_ml', 'trial', 'date', 'day_diff', 'cell_line', 'rapa_nM']]
df_ = df_.append(pd.read_csv('../../data/screen_followups/20220209_proliferation_curves.csv'),
                 ignore_index = True)



df__ = df_#[df_.rapa_nM == 0.0]
df__ = df__[df__.day_diff >= 4]
df__ = df__[df__.cell_line != 'sgTSC1']
df__ = df__[df__.cell_line != 'sgRICTOR']
df__ = df__[df__.trial >= 2]



#############################
#############################
# growth phenotype
#############################
#############################
# day_dict = dict(zip(df_growth.date.unique(), np.arange(7)))
cell_line_dict = dict(zip(df_.cell_line.unique(), colors))
marker_dict = {0: '-', 10: '--', 100: '-.'}

for g,d in df__[df__.cell_line == 'sgCtrl1'].groupby(['day_diff', 'rapa_nM']):
    t = d.day_diff.unique()

    color = cell_lines_colors['sgCtrl1'] #'#B8BABC'

    dens_0 = []
    dens = []
    for trial,_d in d.groupby('trial'):
        d_temp = df__[df__.cell_line == 'sgCtrl1']
        d_temp = d_temp[d_temp.day_diff == 4]
        d_temp = d_temp[d_temp.trial == trial]
        d_temp = d_temp.effective_density_1E6_ml.mean()
        dens = np.append(dens, _d.effective_density_1E6_ml.mean()/d_temp)

    ax_Fi.errorbar(t, np.mean(dens),
                yerr = np.std(dens),
                label = g[1],
                markersize = 7, marker = cell_lines_marker_dict['sgCtrl1'], color  = color,
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1)

d_0 = df__[df__.cell_line == 'sgCtrl1'][df__.day_diff == 4].groupby('trial').effective_density_1E6_ml.mean()
for g,d in df__.groupby(['cell_line', 'rapa_nM']):
    if g[0] != 'sgCtrl1':
        continue
    lw = 1
    dens = []
    t = []
    for day,_d in d.groupby('day_diff'):
        t = np.append(t, day)
        den_m = np.mean(_d.groupby('trial').effective_density_1E6_ml.mean()/d_0)
        dens = np.append(dens, den_m)
    if g[1] >0:
        dens = np.append([1], dens)
        t = np.append(4,t)
    ax_Fi.plot(t[:5], dens[:5], ls = marker_dict[g[1]],
               color = cell_lines_colors['sgCtrl1'],#'#B8BABC',
               label = None, lw = lw)


###############################################

for g,d in df__[df__.cell_line == 'sgLAMTOR1'].groupby(['day_diff', 'rapa_nM']):
    t = d.day_diff.unique()
    color = cell_lines_colors['sgLAMTOR1']#'#738FC1'

    dens_0 = []
    dens = []
    for trial,_d in d.groupby('trial'):
        d_temp = df__[df__.cell_line == 'sgLAMTOR1']
        d_temp = d_temp[d_temp.day_diff == 4]
        d_temp = d_temp[d_temp.trial == trial]
        d_temp = d_temp.effective_density_1E6_ml.mean()
        dens = np.append(dens, _d.effective_density_1E6_ml.mean()/d_temp)

    ax_Fii.errorbar(t, np.mean(dens),
                yerr = np.std(dens),
                label = g[1],
                markersize = 7, marker = cell_lines_marker_dict['sgLAMTOR1'], color  = color,
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1)


d_0 = df__[df__.cell_line == 'sgLAMTOR1'][df__.day_diff == 4].groupby('trial').effective_density_1E6_ml.mean()
for g,d in df__.groupby(['cell_line', 'rapa_nM']):
    if g[0] != 'sgLAMTOR1':
        continue
    lw = 1
    dens = []
    t = []
    for day,_d in d.groupby('day_diff'):
        t = np.append(t, day)
        den_m = np.mean(_d.groupby('trial').effective_density_1E6_ml.mean()/d_0)
        dens = np.append(dens, den_m)
    if g[1] >0:
        dens = np.append([1], dens)
        t = np.append(4,t)
    ax_Fii.plot(t, dens, ls = marker_dict[g[1]],
               color = cell_lines_colors['sgLAMTOR1'], label = None, lw = lw)


###############################################

for g,d in df__[df__.cell_line == 'sgFLCN'].groupby(['day_diff', 'rapa_nM']):
    t = d.day_diff.unique()

    color = cell_lines_colors['sgFLCN']

    dens_0 = []
    dens = []
    for trial,_d in d.groupby('trial'):
        d_temp = df__[df__.cell_line == 'sgFLCN']
        d_temp = d_temp[d_temp.day_diff == 4]
        d_temp = d_temp[d_temp.trial == trial]
        d_temp = d_temp.effective_density_1E6_ml.mean()
        dens = np.append(dens, _d.effective_density_1E6_ml.mean()/d_temp)

    #     yerr = np.abs((d.groupby('date').effective_density_1E6_ml.mean()/d_temp)*np.sqrt((d['effective_density_1E6_ml'].std()/d['effective_density_1E6_ml'].mean())**2 + (d_temp['effective_density_1E6_ml'].std()/d_temp['effective_density_1E6_ml'].mean())**2)
    ax_Fiii.errorbar(t, np.mean(dens),
                yerr = np.std(dens),
                label = g[1],
                markersize = 7, marker = cell_lines_marker_dict['sgFLCN'], color  = color,
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5, lw = 1)


d_0 = df__[df__.cell_line == 'sgFLCN'][df__.day_diff == 4].groupby('trial').effective_density_1E6_ml.mean()
for g,d in df__.groupby(['cell_line', 'rapa_nM']):
    if g[0] != 'sgFLCN':
        continue
    lw = 1
    dens = []
    t = []
    for day,_d in d.groupby('day_diff'):
        t = np.append(t, day)
        den_m = np.mean(_d.groupby('trial').effective_density_1E6_ml.mean()/d_0)
        dens = np.append(dens, den_m)
    if g[1] >0:
        dens = np.append([1], dens)
        t = np.append(4,t)
    ax_Fiii.plot(t, dens, ls = marker_dict[g[1]],
               markersize = 7, color = cell_lines_colors['sgFLCN'],
               label = None, lw = lw)

for _ax in [ax_Fi, ax_Fii, ax_Fiii]:
    _ax.set_xlabel('time post\ndifferentiation [day]')
    _ax.set_xticks([4,6,8])
ax_Fi.set_ylabel('dHL-60 relative\ncell density')

for _ax in [ax_Fii, ax_Fiii]:
    _ax.set_yticks([])

plt.tight_layout()
fig.savefig('../../figures/Fig3F_rapamycin.pdf', bbox_inches='tight')
