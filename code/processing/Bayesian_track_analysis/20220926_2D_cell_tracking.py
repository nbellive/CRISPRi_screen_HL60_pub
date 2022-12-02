import glob
import os
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import convolve2d

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# %matplotlib inline



ip = 200.0/303.0 # Rio 20x air

#

# import  seaborn  as sns

################################
################################
# info
# Here I'll run the Metzner et al. algorithm
# on the 2D data to infer temporal and time-averaged
# persistence values.
################################
################################

################################
################################
# Import cell position/track data
################################
################################

files = glob.glob('../../../data/processed_tracking/202*2D/*')

df_2d = pd.DataFrame()

# Loop through files; grab only relevant 2D files
for f in files:
    df_temp = pd.read_csv(f)
    if 'celltype_x' in  df_temp.columns:
        df_temp = df_temp.rename(columns={"celltype_x": "celltype"})
    if 'sampleset' in  df_temp.columns:
        df_temp = df_temp.rename(columns={"sampleset": "trial"})
    if 'date' not in df_temp.columns:
        df_temp['date'] = f.split('/')[-1][:8]
    if 'trial' not in df_temp.columns:
        df_temp['trial'] = 1
    if 'misc' not in df_temp.columns:
        df_temp['misc'] = ''

    df_2d = df_2d.append(df_temp,  ignore_index = True)

# note that I names some of the lines a little different initially; let's
# make everything consistent.
df_2d = df_2d.replace({'sgCtrl1':'HL-60KW_SC575_sgControl1', 'sgFLCN':'HL-60KW_SC575_sgFLCN',
                       'sgLAMTOR1': 'HL-60KW_SC575_sgLAMTOR1', 'sgGIT2':'HL-60KW_SC575_sgGIT2'})

################################
################################
# to remove  from analysis beacause cell
# was not adhered to coverslip
df_ignore = pd.DataFrame()
################################
################################

# ITGB2:
# 20220215, trial 1, position 1,
# cell 2,7, 9, 11, 17, 19, 22, 25
data_list = {'celltype' : 'HL-60KW_SC575_sgITGB2',
             'date' : 20220215,
            'trial' : 1,
            'position' : 1,
            'cells_ignore' : [2,7, 9, 11, 17, 19, 22, 25]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 2
# cell 1, 17, 14, 3, 16, 11
data_list = {'celltype' : 'HL-60KW_SC575_sgITGB2',
             'date' : 20220215,
            'trial' : 1,
            'position' : 2,
            'cells_ignore' : [1, 17, 14, 3, 16, 11]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 4
# cell 2, 17, 9, 16, 6, 11, 12
data_list = {'celltype' : 'HL-60KW_SC575_sgITGB2',
             'date' : 20220215,
            'trial' : 1,
            'position' : 4,
            'cells_ignore' : [2, 17, 9, 16, 6, 11, 12]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 5
# cell 0, 1, 17,
data_list = {'celltype' : 'HL-60KW_SC575_sgITGB2',
             'date' : 20220215,
            'trial' : 1,
            'position' : 5,
            'cells_ignore' : [0, 1, 17]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 2, position 1,
# cell 0, 4, 10, 5
data_list = {'celltype' : 'HL-60KW_SC575_sgITGB2',
             'date' : 20220215,
            'trial' : 2,
            'position' : 1,
            'cells_ignore' : [0, 4, 10, 5]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# Note that in some videos, there were cells floating in the field of view;
# I've also gone through and identified these so that I can exclude them.

##############################
##############################
# Ctrl1
# 20211209, trial 2, position 2
# cell 118
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20211209,
            'trial' : 2,
            'position' : 2,
            'cells_ignore' : [118]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20211209, trial 2, position 3
# cell 131
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20211209,
            'trial' : 2,
            'position' : 3,
            'cells_ignore' : [131]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208, trial 1, position 1
# cell 42, 46, 45, 41, 36
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220208,
            'trial' : 1,
            'position' : 1,
            'cells_ignore' : [42, 46, 45, 41, 36]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208, trial 1, position 2
# cell 40, 44
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220208,
            'trial' : 1,
            'position' : 2,
            'cells_ignore' : [40, 44]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208, trial 1, position 3
# cell 37, 41
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220208,
            'trial' : 1,
            'position' : 3,
            'cells_ignore' : [37, 41]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 0
# cell 39, 37
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220215,
            'trial' : 1,
            'position' : 0,
            'cells_ignore' : [39, 37]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 1
# cell 49, 38, 42, 36
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220215,
            'trial' : 1,
            'position' :  1,
            'cells_ignore' : [49, 38, 42, 36]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 2
# cell 36, 40
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220215,
            'trial' : 1,
            'position' :  2,
            'cells_ignore' : [36, 40]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215, trial 1, position 3
# cell 42
data_list = {'celltype' : 'HL-60KW_SC575_sgControl1',
             'date' : 20220215,
            'trial' : 1,
            'position' :  3,
            'cells_ignore' : [42]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

##############################
##############################
# 'HL-60KW_SC575_sgFLCN'
# fine!

##############################
##############################
# 'HL-60KW_SC575_sgLAMTOR1'
# 20220208.0, trial 1.0, position 3.0
# cell 30
data_list = {'celltype' : 'HL-60KW_SC575_sgLAMTOR1',
             'date' : 20220208,
            'trial' : 1,
            'position' :  3,
            'cells_ignore' : [30]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208.0, trial 2.0, position 0.0
# cell 3, 15, 7
data_list = {'celltype' : 'HL-60KW_SC575_sgLAMTOR1',
             'date' : 20220208,
            'trial' : 2,
            'position' :  0,
            'cells_ignore' : [3, 15, 7]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208.0, trial 2.0, position 2.0
# cell 63
data_list = {'celltype' : 'HL-60KW_SC575_sgLAMTOR1',
             'date' : 20220208,
            'trial' : 2,
            'position' :  2,
            'cells_ignore' : [63]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220208.0, trial 2.0, position 4.0
# cell 33
data_list = {'celltype' : 'HL-60KW_SC575_sgLAMTOR1',
             'date' : 20220208,
            'trial' : 2,
            'position' :  4,
            'cells_ignore' : [33]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)

# 20220215.0, trial 1.0, position 3.0
# cell 30, 31
data_list = {'celltype' : 'HL-60KW_SC575_sgLAMTOR1',
             'date' : 20220215,
            'trial' : 1,
            'position' :  3,
            'cells_ignore' : [30, 31]}
df_ignore = df_ignore.append(data_list,  ignore_index = True)


################################
# remove cells identified above from DataFrame
df_2d_filtered = pd.DataFrame()

for sg, d in df_2d.groupby(['celltype', 'date', 'trial', 'position']):
    d_ig = df_ignore[(df_ignore['celltype'] == sg[0]) &  (df_ignore['date'] == sg[1]) &  (df_ignore['trial'] == sg[2]) &  (df_ignore['position'] == sg[3])]
    if len(d_ig.cells_ignore) > 0:
        d = d[~d['cell'].isin(d_ig.cells_ignore.values[0])]
    df_2d_filtered = df_2d_filtered.append(d, ignore_index = True)


################################
################################
# Bayesian inference step
################################
################################

# code from Metzner et al.
############################


# compute likelihood on parameter grid
def compLike(vp,v):
    return np.exp(-((v[0] - qGrid*vp[0])**2 + (v[1] - qGrid*vp[1])**2)/(2*a2Grid) - np.log(2*np.pi*a2Grid))

# compute new prior
def compNewPrior(oldPrior,like):
    # compute posterior distribution
    post = oldPrior*like
    post /= np.sum(post)

    # use posterior as a starting point to create new prior
    newPrior = post

    # introduce minimal probability
    mask = newPrior < pMin
    newPrior[mask] = pMin

    # apply boxcar filter
    ker = np.ones((2*Rq + 1, 2*Ra+1))/((2*Rq+1)*(2*Ra+1))

    newPrior = convolve2d(newPrior, ker, mode='same', boundary='symm')

    return newPrior

# compute sequence of posterior distributions for a sequence of measured velocities
def compPostSequ(uList):
    # initialize array for posterior distributions
    dist = np.empty((len(uList),gridSize,gridSize))

    # initialize flat prior
    dist[0].fill(1.0/(gridSize**2))

    # forward pass (create forward priors for all time steps)
    for i in np.arange(1,len(uList)):
        dist[i] = compNewPrior(dist[i-1], compLike(uList[i-1], uList[i]))

    # backward pass
    backwardPrior = np.ones((gridSize,gridSize))/(gridSize**2)
    for i in np.arange(1,len(uList))[::-1]:
        # re-compute likelihood
        like = compLike(uList[i-1], uList[i])

        # forward prior * likelihood * backward prior
        dist[i] = dist[i-1]*like*backwardPrior
        dist[i] /= np.sum(dist[i])

        # generate new backward prior for next iteration
        backwardPrior = compNewPrior(backwardPrior, compLike(uList[i-1], uList[i]))

    # drop initial flat prior before return
    return dist[1:]

# compute posterior mean values from a list of posterior distributions
def compPostMean(postSequ):
    qMean = [np.sum(post*qGrid) for post in postSequ]
    aMean = [np.sum(post*aGrid) for post in postSequ]

    return np.array([qMean,aMean])


# parameter boundaries
gridSize = 200
qBound = [-1.5, 1.5]
aBound = [0.0, 0.5]

# algorithm parameters
pMin = 1.0*10**(-5)
Rq   = 2
Ra   = 2

# likelihood function
# parameter grid (excluding boundary values)
qGrid  = (np.array([np.linspace(qBound[0], qBound[1], gridSize+2)[1:-1]]*gridSize)).T
aGrid  = (np.array([np.linspace(aBound[0], aBound[1], gridSize+2)[1:-1]]*gridSize))
a2Grid = (np.array([np.linspace(aBound[0], aBound[1], gridSize+2)[1:-1]]*gridSize))**2

# Now lets run through the data!
#####################
N_celltypes = len(df_2d_filtered.celltype.unique())
celltype_index = dict(zip(df_2d_filtered.celltype.sort_values().unique(), np.arange(0,N_celltypes).T))

for i, data in df_2d_filtered.groupby(['celltype']):
    print(i)

    # make DataFrame to save Parameter estimates
    df_Bayesian = pd.DataFrame()
    df_Bayesian2 = pd.DataFrame()

    activity_avg = []
    persistence_avg = []

    index = celltype_index[i]
    material = i

    vel = np.array([])
    count = 0

    meanPost1_all_P = []
    meanPost1_all_A = []
    meanPost1_length = []

    for j, data_ in data.groupby(['cell', 'trial', 'position', 'date']):

        data_ = data_[['x','y', 'frame']]
        dx = []
        dy = []
        dz = []

        df = data_
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        disp = 0
        series_len = df.frame.max() - df.frame.min()

        if df.empty:
            continue

        for t in np.arange(df.frame.min(),df.frame.max()):

            dx_ = df[df.frame==t+1].x.values - \
                    df[df.frame==t].x.values

            dx = np.append(dx,dx_)

            dy_ = df[df.frame==t+1].y.values - \
                    df[df.frame==t].y.values
            dy = np.append(dy,dy_)


            if (len(dx_) == 0):
                continue

        if len(dx)<15:
            continue

        # Now lets begin Bayesian analysis of cell track
        veloSequ1 = np.zeros([len(dx), 2])
        veloSequ1[:,0] = dx/30.0 # um /sec
        veloSequ1[:,1] = dy/30.0 # um /sec

        postSequ1 = compPostSequ(veloSequ1)
#         print(postSequ1)
        meanPost1 = compPostMean(postSequ1)
        meanPost1_all_P = np.append(meanPost1_all_P, meanPost1[0])
        meanPost1_all_A = np.append(meanPost1_all_A, meanPost1[1])
        count += 1

        # append average persistence and activity values to dataframe
        data_Bayesian = {'cell':j[0],
                'celltype':i,
                'trial':j[1],
                'date':j[-1],
                'position':j[-2],
                 'concentration' : '',
                'average_persistence':np.mean(meanPost1[0]),
                'activity ($\mu$m/sec)':np.mean(meanPost1[1]),
                'speed ($\mu$m/sec)': np.mean(np.sqrt(veloSequ1[:,0]**2 + \
                                    veloSequ1[:,1]**2)),
                         'material':material}
        df_Bayesian = df_Bayesian.append(data_Bayesian, ignore_index=True)

        # append all inference values to dataframe
        for t in np.arange(len(veloSequ1)-1):#df.frame.unique():#np.arange(df.frame.min(), df.frame.max()):# + len(dx)):
            data_Bayesian2 = {'cell':j[0],
                              'celltype':i,
                              'frame':t,
                'Vx ($\mu$m/sec)':veloSequ1[:,0][int(t)],
                'Vy ($\mu$m/sec)':veloSequ1[:,1][int(t)],
                'speed ($\mu$m/sec)': np.sqrt(veloSequ1[:,0][int(t)]**2 + \
                                    veloSequ1[:,1][int(t)]**2 ),
                'trial':j[1],
                'date':j[-1],
                'position':j[-2],
                 'concentration' : '',
                'material':material,
                'persistence':meanPost1[0][int(t)],
                'activity ($\mu$m/sec)':meanPost1[1][int(t)]}
            df_Bayesian2 = df_Bayesian2.append(data_Bayesian2, ignore_index=True)

    fname_ = '_'.join([i, '.csv'])
    df_Bayesian.to_csv('../../../data/processed_tracking_bayesian/20220920_2D_filtered_avg_'+fname_)
    df_Bayesian2.to_csv('../../../data/processed_tracking_bayesian/20220920_2D_filtered_'+fname_)
