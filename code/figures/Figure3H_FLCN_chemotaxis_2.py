import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

import numpy as np
import glob
import pandas as pd
import math
import numpy as np

import seaborn as sns
import scipy.io as spio

plt.style.use('styleNB.mplstyle')

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
#  Figure 7: Understanding the changes in candidates like LAMTOR1 and FLCN
# (B) schematic of RNA-seq experiment and UMAP result
#############################################

fig = plt.figure(figsize=(3,4))
gs = GridSpec(nrows=2, ncols=3, height_ratios=[0.75, 1], width_ratios=[1, 1, 1])

ax_Fi    = fig.add_subplot(gs[1,0])
ax_Fii   = fig.add_subplot(gs[1,1])
ax_Fiii  = fig.add_subplot(gs[1,2])

###########################
###########################
# Violin plots
###########################
###########################

# load in data
df = pd.read_csv('../../data/processed_tracking/20220404_chemotaxis/20220404_chemotaxis_compiled_data_Matlab_to_DataFrame.csv')
###########################
# s0 - basal speed
###########################
y = df[df.rowlabels.str.contains('sgCONTROL-')].s0/60 #per second
s0_ctrl = ax_Fi.violinplot(y,
                    positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in s0_ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_Fi.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('sgCONTROL-')].s0/60
    ax_Fi.errorbar(np.random.normal(0.5, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)



y = df[df.rowlabels.str.contains('FLCN')].s0/60
s0_flcn = ax_Fi.violinplot(y,
                    positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in s0_flcn['bodies']:
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.5)

ax_Fi.hlines(y.median(), 1-0.2,1+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('FLCN')].s0/60
    ax_Fi.errorbar(np.random.normal(1, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

ax_Fi.set_ylim(0,0.3)

###########################
# a1 - avg angular displacement
###########################
y = df[df.rowlabels.str.contains('sgCONTROL-')].a1
s0_ctrl = ax_Fii.violinplot(y,
                    positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in s0_ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_Fii.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('sgCONTROL-')].a1
    ax_Fii.errorbar(np.random.normal(0.5, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)



y = df[df.rowlabels.str.contains('FLCN')].a1
a1_flcn = ax_Fii.violinplot(y,
                    positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in a1_flcn['bodies']:
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.5)

ax_Fii.hlines(y.median(), 1-0.2,1+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('FLCN')].a1
    ax_Fii.errorbar(np.random.normal(1, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

ax_Fii.set_ylim(0,30)

###########################
# a1 - avg angular displacement
###########################
y = df[df.rowlabels.str.contains('sgCONTROL-')].c1
s0_ctrl = ax_Fiii.violinplot(y,
                    positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in s0_ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_Fiii.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('sgCONTROL-')].c1
    ax_Fiii.errorbar(np.random.normal(0.5, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)



y = df[df.rowlabels.str.contains('FLCN')].c1
a1_flcn = ax_Fiii.violinplot(y,
                    positions = [1], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in a1_flcn['bodies']:
    pc.set_color(cell_lines_colors['sgFLCN'])
    pc.set_alpha(0.5)

ax_Fiii.hlines(y.median(), 1-0.2,1+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby(['date', 'rowlabels']):
    y_i = d[d.rowlabels.str.contains('FLCN')].c1
    ax_Fiii.errorbar(np.random.normal(1, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

ax_Fiii.set_ylim(0,4)

plt.tight_layout()
fig.savefig('../../figures/Fig4CD_FLCN_chemotaxis_2.pdf', bbox_inches='tight')
