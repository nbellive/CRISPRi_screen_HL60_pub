import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

import numpy as np
import glob
import pandas as pd
import math

import numpy as np
from sklearn.preprocessing import StandardScaler
import skimage
import skimage.io
from skimage.morphology import disk
from skimage.filters import rank

from scipy.interpolate import make_interp_spline, BSpline
from scipy import interpolate

import seaborn as sns
import scipy.io as spio

import math

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle_fcn(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def angle_atan2(v1, v2):
    return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])


# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict




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

ax_Bi    = fig.add_subplot(gs[0,:2], projection='polar')

ax_Fi    = fig.add_subplot(gs[1,0])
ax_Fii   = fig.add_subplot(gs[1,1])
ax_Fiii  = fig.add_subplot(gs[1,2])


###################################
# rose plots
###################################

files_ = glob.glob('../../data/processed_tracking/20220404_chemotaxis/compiled_fMLP_tracking_*.csv')
df_track = pd.DataFrame()
for f in files_:
    df_temp = pd.read_csv(f)
    df_track = df_track.append(df_temp)

# location where intensity of fMLP is at maximum
ind = [512, 512]

# sgControl line
angle_test = []
angle_test_2 = []
for well, df in df_track[df_track.celltype == 'sgCONTROL-NEW'].groupby(['date','well']):
    df = df[df.frame >= 15]
#     df = df[df.frame >= 30]
    num_cells = len(df.cell.unique())
    vel = []
    dx = []
    dy = []
    dx_grad = []
    dy_grad = []


    for i in df.cell.unique():
        data_ = df[df.cell==i]

        ###################################################
        ########## first and last. frame (dx,dy) ########
        ###################################################
        dx_ = data_[data_.frame==data_.frame.max()].x.values - \
                data_[data_.frame==data_.frame.min()].x.values
        dx = np.append(dx,dx_)

        dy_ = data_[data_.frame==data_.frame.max()].y.values - \
                data_[data_.frame==data_.frame.min()].y.values
        dy = np.append(dy,dy_)

        ###################################################
        ########## first and last. frame (dx,dy) ########
        ###################################################
        dx_2 = ind[1] - \
                data_[data_.frame==data_.frame.min()].x.values

        dy_2 = ind[0] - \
                data_[data_.frame==data_.frame.min()].y.values

        angle_ = angle_fcn([dx_,dy_], [dx_2,dy_2])
        angle_test = np.append(angle_test, angle_)
        angle_2 = angle_atan2([dx_,dy_], [dx_2,dy_2])
        angle_test_2 = np.append(angle_test_2, angle_2)

# # plot rose plot

angle = angle_test_2 * 180 / np.pi
angle_p =  angle[angle>=0]
angle_m =  angle[angle<=0]
angle = np.append(angle_p,angle_m+360)

angle_p =  angle[angle<=180]
angle_m =  angle[angle>=180]
angle = np.append(angle_p,360-angle_m)

degrees = angle #+ 180# np.random.randint(0, 360, size=200)
radians = np.deg2rad(angle_test)

bin_size = 30
a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])


ax_Bi.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0,
        color='#B8BABC', edgecolor='k', linewidth = 0.5)#, zorder = 10)
ax_Bi.set_theta_zero_location("E")
ax_Bi.set_theta_direction(1)
ax_Bi.set_yticklabels([])
# ax_Bi.set_rmax(100)
# ax_Bi.grid(False)
ax_Bi.tick_params(labelsize=14)

###########################
# FLCN

angle_test = []
angle_test_2 = []
for well, df in df_track[df_track.celltype == 'sgFLCN'].groupby(['date','well']):
    df = df[df.frame >= 15]
    num_cells = len(df.cell.unique())
    vel = []
    dx = []
    dy = []
    dx_grad = []
    dy_grad = []


    for i in df.cell.unique():
        data_ = df[df.cell==i]

        ###################################################
        ########## first and last. frame (dx,dy) ########
        ###################################################
        dx_ = data_[data_.frame==data_.frame.max()].x.values - \
                data_[data_.frame==data_.frame.min()].x.values
        dx = np.append(dx,dx_)

        dy_ = data_[data_.frame==data_.frame.max()].y.values - \
                data_[data_.frame==data_.frame.min()].y.values
        dy = np.append(dy,dy_)

        ###################################################
        ########## first and last. frame (dx,dy) ########
        ###################################################
        dx_2 = ind[1] - \
                data_[data_.frame==data_.frame.min()].x.values
#         dx_grad = np.append(dx_grad,dx_2)

        dy_2 = ind[0] - \
                data_[data_.frame==data_.frame.min()].y.values
#         dy_grad = np.append(dy_grad,dy_2)

        angle_ = angle_fcn([dx_,dy_], [dx_2,dy_2])
        angle_test = np.append(angle_test, angle_)
        angle_2 = angle_atan2([dx_,dy_], [dx_2,dy_2])
        angle_test_2 = np.append(angle_test_2, angle_2)

# deg_sym = np.rad2deg(angle_test)

angle = angle_test_2 * 180 / np.pi
angle_p =  angle[angle>=0]
angle_m =  angle[angle<=0]
angle = np.append(angle_p,angle_m+360)

angle_p =  angle[angle<=180]
angle_m =  angle[angle>=180]
angle = np.append(angle_p,360-angle_m)

degrees = angle #+ 180# np.random.randint(0, 360, size=200)
radians = np.deg2rad(angle_test)

bin_size = 30
a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))

centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

ax_Bi.bar(centers[::-1], a, width=np.deg2rad(bin_size),  linewidth = 0.5, bottom=0.0,
            color = cell_lines_colors['sgFLCN'], edgecolor='k')#, zorder = 10)
ax_Bi.set_theta_zero_location("E")
ax_Bi.set_theta_direction(1)
ax_Bi.set_yticklabels([])
# ax_Bii.grid(False)
ax_Bi.tick_params(labelsize=10)




###########################
###########################
# Violin plots
###########################
###########################
# files = glob.glob('../../../../../../../Volumes/Belliveau_RAW_3_JTgroup/2022_CollinsLab_HTChemotaxis-Nathan-TheriotLab-5dayDiff/mat files_updated/*/processed*.mat')
files = glob.glob('../../../../../../../Volumes/ExpansionHomesA/nbelliveau/microscopy_raw/2022_CollinsLab_HTChemotaxis-Nathan-TheriotLab-5dayDiff/mat files_updated/*/processed*.mat')

df = pd.DataFrame()
for f in files:
    vig=loadmat(f)
    df_temp = pd.DataFrame.from_dict(vig['mer1'])
    df_temp['date'] = f.split('/')[-2][:10]
    df = df.append(df_temp, ignore_index = True)

###########################
# s0 - basal speed
###########################
y = df[df.rowlabels.str.contains('sgCONTROL-')].s0/60
s0_ctrl = ax_Fi.violinplot(y,
                    positions = [0.5], points=60, widths=0.3,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.2)

for pc in s0_ctrl['bodies']:
    pc.set_color('#B8BABC')
    pc.set_alpha(0.5)

ax_Fi.hlines(y.median(), 0.5-0.2,0.5+0.2, zorder=10, lw = 1.5)

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby('date'):
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

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby('date'):
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

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby('date'):
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

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby('date'):
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

for date, d in df[df.rowlabels.str.contains('sgCONTROL-')].groupby('date'):
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

for date, d in df[df.rowlabels.str.contains('FLCN')].groupby('date'):
    y_i = d[d.rowlabels.str.contains('FLCN')].c1
    ax_Fiii.errorbar(np.random.normal(1, 0.05, 1), y_i.median(),
                markersize = 4, marker = 'o', color  = 'k',
                    markeredgecolor = 'k',
                    markeredgewidth = 0.5,
                   lw = 0.5,
                  alpha = 0.3)

ax_Fiii.set_ylim(0,4)

plt.tight_layout()
fig.savefig('../../figures/Fig3F_FLCN_chemotaxis.pdf', bbox_inches='tight')
