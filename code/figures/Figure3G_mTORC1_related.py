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
import scipy
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import seaborn as sns

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.05, 1.0), loc='upper left')


plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']

colors2 = sns.color_palette("Set2")

cell_lines_colors = {'sgControl' : '#B8BABC',
  'sgCtrl1' : '#B8BABC',
  'sgFLCN': colors2[3],
  'sgLAMTOR1':   sns.color_palette("husl", 8)[6],
  'sgSPI1' :  colors2[0],
  'sgTSC1' :  '#738FC1',
  'sgRICTOR' : '#738FC1',
  'sgCEBPE' : '#738FC1'}

#############################################
#  Figure 3: Characterization of cells following KD of LAMTOR1 and FLCN
# (A, B, C) Characterization of differentiation markers (different .py file)
# (D) RNA-seq analysis
# (E) rapamycin treatment
# (F) phase images of sgFLCN, sgLAMTOR1, and sgControl1 (added in w Illustrator)
# (G) migration analysis
# (H) chemotaxis of FLCN
#############################################

#############################
#############################
# (E) cell migration experiments, 2D/3D
#############################
#############################

fig = plt.figure(figsize=(3,3))
gs = GridSpec(nrows=2, ncols=2,
              height_ratios=[1, 1],
              width_ratios=[1, 1])

ax_2d = fig.add_subplot(gs[0, 0])
ax_3d = fig.add_subplot(gs[0,1])

ax_2d_p = fig.add_subplot(gs[1,0])
ax_3d_p = fig.add_subplot(gs[1,1])

files_bay_2d = glob.glob('../../data/processed_tracking_bayesian/20220328_2D_filtered_avg*all*')

df_Bayes_2d = pd.DataFrame()
for f in files_bay_2d:
    df_temp = pd.read_csv(f)
    df_Bayes_2d = df_Bayes_2d.append(df_temp, ignore_index = True)
df_Bayes_2d = df_Bayes_2d[['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'speed ($\mu$m/sec)', 'trial']]
df_Bayes_2d.columns = ['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'average_speed', 'trial']
df_Bayes_2d = df_Bayes_2d[df_Bayes_2d.celltype.isin(['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgFLCN', 'HL-60KW_SC575_sgLAMTOR1'])]

files_bay_3d = glob.glob('../../data/processed_tracking_bayesian/20220328_3D_filtered_*avg*all*')

df_Bayes_3d = pd.DataFrame()
for f in files_bay_3d:
    df_temp = pd.read_csv(f)
    df_Bayes_3d = df_Bayes_3d.append(df_temp, ignore_index = True)
df_Bayes_3d = df_Bayes_3d[['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'speed ($\mu$m/sec)', 'trial']]
df_Bayes_3d.columns = ['average_persistence', 'cell', 'celltype', 'concentration',
                'date', 'material', 'position', 'average_speed', 'trial']
df_Bayes_3d = df_Bayes_3d[df_Bayes_3d.celltype.isin(['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgFLCN', 'HL-60KW_SC575_sgLAMTOR1'])]
df_Bayes_3d = df_Bayes_3d[df_Bayes_3d.concentration == '0.75mgml']

# We also need to load in the processed tracking data.
# To deal with the 3D data in a similar way as we've done for the 2D, we're
# going to pick multiple positions; in this case, we're simply going
# to take position 1 as the first 100 um, and position 2 as the top 100 um.
# I think this is worthwhile in case there are some spatial differences
# in the collagen matrix; it also reduces the number of cells per position
# that we're averaging across.
files = glob.glob('../../data/processed_tracking/2022*3D/*')

df_3d = pd.DataFrame()
for f in files:
    if 'Thresh' in f:
        continue
    if 'rapa' in f:
        continue

    df_temp = pd.read_csv(f)

    if 'celltype_x' in df_temp.columns:
        if df_temp.celltype_x.unique() == 'HL-60KW_SC575_sgCtrl1':
            df_temp.celltype_x = 'HL-60KW_SC575_sgControl1'
        df_temp = df_temp.rename(columns={"celltype_x": "celltype"})
    if 'date' not in df_temp.columns:
        df_temp['date'] = f.split('/')[-1][:8]
    if 'misc' not in df_temp.columns:
        df_temp['misc'] = ''
    df_3d = df_3d.append(df_temp,  ignore_index = True)

df_3d = df_3d[df_3d.concentration == '0.75mgml']

#############################
# speed
#############################
########
# 2D
########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgControl1']['average_speed']
vp_2d_Ctrl = ax_2d.violinplot(y,positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_Ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_2d.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)
print('2D, sgControl1: ', y.mean())

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgControl1'].groupby(['date', 'trial', 'position']):
    ax_2d.errorbar(np.random.normal(0.5, 0.05, 1), d['average_speed'].median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgFLCN']['average_speed']
vp_2d_FLCN = ax_2d.violinplot(y,positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_FLCN['bodies']:
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.6)

ax_2d.hlines(y.mean(), 1-0.2,1+0.2, zorder=10, lw = 1.5)
print('2D, FLCN: ', y.mean())

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgFLCN'].groupby(['date', 'trial', 'position']):
    ax_2d.errorbar(np.random.normal(1, 0.05, 1), d['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgLAMTOR1']['average_speed']
vp_2d_LAMTOR1 = ax_2d.violinplot(y,positions = [1.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_LAMTOR1['bodies']:
    pc.set_color(cell_lines_colors['sgLAMTOR1'])
    pc.set_alpha(0.6)

ax_2d.hlines(y.median(), 1.5-0.2,1.5+0.2, zorder=10, lw = 1.5)
print('2D, LAMTOR1: ', y.mean())

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgLAMTOR1'].groupby(['date', 'trial', 'position']):
    d_top = d
    ax_2d.errorbar(np.random.normal(1.5, 0.05, 1), d['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)



########
# 3D
########

y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgControl1']['average_speed']
vp_3d_Ctrl = ax_3d.violinplot(y,positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_Ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_3d.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)
print('3D, sgControl1: ', y.mean())

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgControl1'].groupby(['date', 'trial']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgControl1']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d.errorbar(np.random.normal(0.5, 0.05, 1), d_top['average_speed'].median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d.errorbar(np.random.normal(0.5, 0.05, 1), d_bottom['average_speed'].median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgFLCN']['average_speed']
vp_3d_FLCN = ax_3d.violinplot(y,positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_FLCN['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.6)

ax_3d.hlines(y.mean(), 1-0.2,1+0.2, zorder=10, lw = 1.5)
print('3D, FLCN: ', y.mean())

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgFLCN'].groupby(['date', 'trial']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgFLCN']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d.errorbar(np.random.normal(1, 0.05, 1), d_top['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d.errorbar(np.random.normal(1, 0.05, 1), d_bottom['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1']['average_speed']
vp_3d_LAMTOR1 = ax_3d.violinplot(y,positions = [1.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_LAMTOR1['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgLAMTOR1'])
    pc.set_alpha(0.6)

ax_3d.hlines(y.median(), 1.5-0.2,1.5+0.2, zorder=10, lw = 1.5)
print('3D, LAMTOR1: ', y.mean())

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1'].groupby(['date', 'trial']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d.errorbar(np.random.normal(1.5, 0.05, 1), d_top['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d.errorbar(np.random.normal(1.5, 0.05, 1), d_bottom['average_speed'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)


ax_2d.set_xlim(0,2)
ax_2d.set_ylim(0,0.4)
ax_2d.set_xticks([])

ax_3d.set_xlim(0,2)
ax_3d.set_ylim(0,0.2)
ax_3d.set_xticks([])

#############################
# persistence
#############################

########
# 2D
########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgControl1']['average_persistence']
vp_2d_Ctrl = ax_2d_p.violinplot(y,positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_Ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_2d_p.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgControl1'].groupby(['date', 'trial', 'position']):
    ax_2d_p.errorbar(np.random.normal(0.5, 0.05, 1), d['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgFLCN']['average_persistence']
vp_2d_FLCN = ax_2d_p.violinplot(y,positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_FLCN['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.6)

ax_2d_p.hlines(y.mean(), 1-0.2,1+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgFLCN'].groupby(['date', 'trial', 'position']):
    ax_2d_p.errorbar(np.random.normal(1, 0.05, 1), d['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgLAMTOR1']['average_persistence']
vp_2d_LAMTOR1 = ax_2d_p.violinplot(y,positions = [1.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_2d_LAMTOR1['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgLAMTOR1'])
    pc.set_alpha(0.6)

ax_2d_p.hlines(y.median(), 1.5-0.2,1.5+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_2d[df_Bayes_2d.celltype == 'HL-60KW_SC575_sgLAMTOR1'].groupby(['date', 'trial', 'position']):
    ax_2d_p.errorbar(np.random.normal(1.5, 0.05, 1), d['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)



########
# 3D
########

y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgControl1']['average_persistence']
vp_3d_Ctrl = ax_3d_p.violinplot(y,positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_Ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_3d_p.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgControl1'].groupby(['date', 'trial']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgControl1']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d_p.errorbar(np.random.normal(0.5, 0.05, 1), d_top['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d_p.errorbar(np.random.normal(0.5, 0.05, 1), d_bottom['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgFLCN']['average_persistence']
vp_3d_FLCN = ax_3d_p.violinplot(y,positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_FLCN['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.6)

ax_3d_p.hlines(y.mean(), 1-0.2,1+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgFLCN'].groupby(['date', 'trial']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgFLCN']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d_p.errorbar(np.random.normal(1, 0.05, 1), d_top['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d_p.errorbar(np.random.normal(1, 0.05, 1), d_bottom['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

########
y = df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1']['average_persistence']
vp_3d_LAMTOR1 = ax_3d_p.violinplot(y,positions = [1.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in vp_3d_LAMTOR1['bodies']:
    # pc.set_color('#D5DDEA')
    pc.set_color(cell_lines_colors['sgLAMTOR1'])
    pc.set_alpha(0.6)

ax_3d_p.hlines(y.median(), 1.5-0.2,1.5+0.2, zorder=10, lw = 1.5)

for g, d in df_Bayes_3d[df_Bayes_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1'].groupby(['date', 'trial', 'position']):
    df_3d_ = df_3d[df_3d.celltype == 'HL-60KW_SC575_sgLAMTOR1']
    df_3d_ = df_3d_[df_3d_.date == g[0]]
    df_3d_ = df_3d_[df_3d_.trial == g[1]]
    df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
    df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

    d_top = d[d.cell.isin(df_3d_top)]
    d_bottom =  d[d.cell.isin(df_3d_bottom)]

    ax_3d_p.errorbar(np.random.normal(1.5, 0.05, 1), d_top['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

    ax_3d_p.errorbar(np.random.normal(1.5, 0.05, 1), d_bottom['average_persistence'].mean(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)


ax_2d_p.set_xlim(0,2)
ax_2d_p.set_ylim(-0.5,1.0)
ax_2d_p.set_xticks([])

ax_3d_p.set_xlim(0,2)
ax_3d_p.set_ylim(-0.5,1.0)
ax_3d_p.set_xticks([])

ax_2d_p.spines['bottom'].set_position(('data', 0))
ax_3d_p.spines['bottom'].set_position(('data', 0))




plt.tight_layout()
fig.savefig('../../figures/Fig3G_mTORCcandidates.pdf')


########################
########################
# statistics
########################
########################


lines_2D = ['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgFLCN', 'HL-60KW_SC575_sgLAMTOR1']
lines_3D = ['HL-60KW_SC575_sgControl1', 'HL-60KW_SC575_sgFLCN', 'HL-60KW_SC575_sgLAMTOR1']

###################
# Speed 2d
###################
print('######################')
print('Speed, 2D')
print('######################')

param = 'average_speed'
exp_means_2d_wt = []
exp_means_2d_flcn = []
exp_means_2d_lam = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_flcn = np.append(exp_means_2d_flcn, d[param].mean())
        elif i == 2:
            exp_means_2d_lam = np.append(exp_means_2d_lam, d[param].mean())

print('2D, speed, flcn', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_flcn))
print('2D, speed, lamtor1', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_lam))


#perform one-way ANOVA
a = exp_means_2d_wt
b = exp_means_2d_flcn
c = exp_means_2d_lam
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'speed' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'FLCN',
                                'speed' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'LAMTOR1',
                                'speed' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['speed'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)

###################
# Speed 3d
###################

exp_means_3d_wt = []
exp_means_3d_flcn = []
exp_means_3d_lam = []

for i, ct in enumerate(lines_3D):

    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        df_3d_ = df_3d[df_3d.celltype == ct]
        df_3d_ = df_3d_[df_3d_.date == g[0]]
        df_3d_ = df_3d_[df_3d_.trial == g[1]]
        df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
        df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

        d_top = d[d.cell.isin(df_3d_top)]
        d_bottom =  d[d.cell.isin(df_3d_bottom)]
        if np.any([d_top.empty, d_bottom.empty]):
            continue
        else:
            if i == 0:
                exp_means_3d_wt = np.append(np.append(exp_means_3d_wt, d_top[param].mean()),d_bottom[param].mean())
            elif i == 1:
                exp_means_3d_flcn = np.append(np.append(exp_means_3d_flcn, d_top[param].mean()),d_bottom[param].mean())
            elif i == 2:
                exp_means_3d_lam = np.append(np.append(exp_means_3d_lam, d_top[param].mean()),d_bottom[param].mean())

print(exp_means_3d_wt, exp_means_3d_flcn)
print('3D, speed, flcn', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_flcn))
print('3D, speed, lamtor1', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_lam))

print('######################')
print('Speed, 3D')
print('######################')


#perform one-way ANOVA
a = exp_means_3d_wt
b = exp_means_3d_flcn
c = exp_means_3d_lam
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'speed' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'FLCN',
                                'speed' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'LAMTOR1',
                                'speed' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['speed'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)

###################
# persistence 2d
###################
print('######################')
print('Persistence, 2D')
print('######################')

param = 'average_persistence'
exp_means_2d_wt = []
exp_means_2d_flcn = []
exp_means_2d_lam = []
for i, ct in enumerate(lines_2D):
    for g, d in df_Bayes_2d[df_Bayes_2d.celltype == ct].groupby(['date', 'trial', 'position']):
        if i == 0:
            exp_means_2d_wt = np.append(exp_means_2d_wt, d[param].mean())
        elif i == 1:
            exp_means_2d_flcn = np.append(exp_means_2d_flcn, d[param].mean())
        elif i == 2:
            exp_means_2d_lam = np.append(exp_means_2d_lam, d[param].mean())

print('2D, persistence, flcn', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_flcn))
print('2D, persistence, lamtor1', scipy.stats.mannwhitneyu(exp_means_2d_wt, exp_means_2d_lam))


#perform one-way ANOVA
a = exp_means_2d_wt
b = exp_means_2d_flcn
c = exp_means_2d_lam
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'persistence' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'FLCN',
                                'persistence' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'LAMTOR1',
                                'persistence' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['persistence'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)

###################
# persistence 3d
###################
print('######################')
print('Persistence, 3D')
print('######################')
exp_means_3d_wt = []
exp_means_3d_flcn = []
exp_means_3d_lam = []
for i, ct in enumerate(lines_3D):

    print(ct)
    for g, d in df_Bayes_3d[df_Bayes_3d.celltype == ct].groupby(['date', 'trial', 'position']):
        df_3d_ = df_3d[df_3d.celltype == ct]
        df_3d_ = df_3d_[df_3d_.date == g[0]]
        df_3d_ = df_3d_[df_3d_.trial == g[1]]
        df_3d_top = df_3d_[df_3d_.z>100].cell.unique()
        df_3d_bottom = df_3d_[df_3d_.z<=100].cell.unique()

        d_top = d[d.cell.isin(df_3d_top)]
        d_bottom =  d[d.cell.isin(df_3d_bottom)]
        if np.any([d_top.empty, d_bottom.empty]):
            continue
        else:
            if i == 0:
                exp_means_3d_wt = np.append(np.append(exp_means_3d_wt, d_top[param].mean()),d_bottom[param].mean())
            elif i == 1:
                exp_means_3d_flcn = np.append(np.append(exp_means_3d_flcn, d_top[param].mean()),d_bottom[param].mean())
            elif i == 2:
                exp_means_3d_lam = np.append(np.append(exp_means_3d_lam, d_top[param].mean()),d_bottom[param].mean())
print(exp_means_3d_wt, exp_means_3d_flcn)
print('3D, persistence, flcn', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_flcn))
print('3D, persistence, lamtor1', scipy.stats.mannwhitneyu(exp_means_3d_wt, exp_means_3d_lam))
print('')


#perform one-way ANOVA
a = exp_means_3d_wt
b = exp_means_3d_flcn
c = exp_means_3d_lam
print('one-way ANOVA', f_oneway(a, b, c))

#create DataFrame to hold data
df_stats = pd.DataFrame()
for val in a:
    df_stats = df_stats.append({'group' : 'WT',
                                'persistence' : val}, ignore_index = True)
for val in b:
    df_stats = df_stats.append({'group' : 'FLCN',
                                'persistence' : val}, ignore_index = True)
for val in c:
    df_stats = df_stats.append({'group' : 'LAMTOR1',
                                'persistence' : val}, ignore_index = True)

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_stats['persistence'],
                          groups=df_stats['group'],
                          alpha=0.05)

#display results
print(tukey)
