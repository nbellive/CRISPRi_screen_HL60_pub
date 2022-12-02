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


ip = 500.0/772.0 # iSIM, xIon 20x air

#

# import  seaborn  as sns

################################
################################
# info
# Here I'll run the Metzner et al. algorithm
# on the 3D data to infer temporal and time-averaged
# persistence values.
################################
################################

################################
################################
# Import cell position/track data
################################
################################

files = glob.glob('../../../data/processed_tracking/2022*3D/*')

df_3d = pd.DataFrame()
for f in files:
    if 'Thresh' in f:
        continue
    if 'rapa' in f:
        continue

    df_temp = pd.read_csv(f)
    print(f)
    print(df_temp.columns)
    if df_temp['concentration'].unique() != '0.75mgml':
        continue
    if 'celltype_x' in df_temp.columns:
        if df_temp.celltype_x.unique() == 'HL-60KW_SC575_sgCtrl1':
            df_temp.celltype_x = 'HL-60KW_SC575_sgControl1'
        df_temp = df_temp.rename(columns={"celltype_x": "celltype"})
    if 'date' not in df_temp.columns:
        df_temp['date'] = f.split('/')[-1][:8]
    if 'misc' not in df_temp.columns:
        df_temp['misc'] = ''
    df_3d = df_3d.append(df_temp,  ignore_index = True)

################################
################################
# Drift correction x, y, z
################################
################################

# determine xy shift for  drift correction
#  Only worry about cells quantified throughout time series.
# generate a dictionary  for x and y

df_filtered = pd.DataFrame()

for sg, d in df_3d[df_3d.frame!=60].groupby(['celltype']):
    print(sg)

    count = 0
    for g, d_ in d.groupby(['concentration', 'misc', 'date', 'trial', 'position']):
        count += 1

    aq = 0
    for g, d_ in d.groupby(['concentration', 'misc', 'date', 'trial', 'position']):
        print(g)

        ##################################
        # identify non-motile cells for drift correction
        ##################################
        num_cells = len(d_.cell.unique())
        nonmotile_cells = []
        for i in np.arange(0,num_cells):
            data_ = d_[d_.cell==i]

            data_ = data_.replace([np.inf, -np.inf], np.nan)
            d_x = data_.x[~np.isnan(data_.x)]
            d_y = data_.y[~np.isnan(data_.y)]
            d_z = data_.z[~np.isnan(data_.z)]

            if np.any(d_z<=30.0):
                continue
            elif np.any(d_z>=170.0):
                continue

            if len(d_x) ==60:
                if np.all([np.ptp(d_x)<=25.0, np.ptp(d_y)<=25.0, np.ptp(d_z)<=10.0]):
                    nonmotile_cells = np.append(nonmotile_cells, i)

        s_x = [0]
        s_y = [0]
        s_z = [0]
        df_nonmotile = d_[d_.cell.isin(nonmotile_cells)]

        for t in np.arange(0,60):
            if t == 0:
                continue
            shift_temp_x = []
            shift_temp_y = []
            shift_temp_z = []
            for cell, d_nm in df_nonmotile.groupby('cell'):
                shift_temp_x = np.append(shift_temp_x, d_nm[d_nm.frame==t].x.values - d_nm[d_nm.frame==t-1].x.values)
                shift_temp_y = np.append(shift_temp_y, d_nm[d_nm.frame==t].y.values - d_nm[d_nm.frame==t-1].y.values)
                shift_temp_z = np.append(shift_temp_z, d_nm[d_nm.frame==t].z.values - d_nm[d_nm.frame==t-1].z.values)

            s_x = np.append(s_x, np.mean(shift_temp_x))
            s_y = np.append(s_y, np.mean(shift_temp_y))
            s_z = np.append(s_z, np.mean(shift_temp_z))


        ##################################
        # Apply drift correction, plot, and save motile cell track data
        ##################################
        num_cells = len(d_.cell.unique())
        for i in np.arange(0,num_cells):
            data_ = d_[d_.cell==i]

            data_ = data_.replace([np.inf, -np.inf], np.nan)
            d_x = data_.x[~np.isnan(data_.x)]
            d_y = data_.y[~np.isnan(data_.y)]
            d_z = data_.z[~np.isnan(data_.z)]

            if len(d_x)==0:
                continue
            if sum(d_z<=10.0)/len(d_z) >= 0.2:
                continue
            elif sum(d_z>=190.0)/len(d_z) >= 0.2:
                continue

            if len(d_x) >=40:
                if len(df_nonmotile) == 0:
                    data_['x_shifted'] = data_['x']
                    data_['y_shifted'] = data_['y']
                    data_['z_shifted'] = data_['z']
                    d_s_x = data_.x_shifted
                    d_s_y = data_.y_shifted
                    d_s_z = data_.z_shifted
                else:
                    x = data_.x.values - np.cumsum(s_x[int(data_.frame.min()):int(data_.frame.max())+1])
                    y = data_.y.values - np.cumsum(s_y[int(data_.frame.min()):int(data_.frame.max())+1])
                    z = data_.z.values - np.cumsum(s_z[int(data_.frame.min()):int(data_.frame.max())+1])
                    data_['x_shifted'] = x
                    data_['y_shifted'] = y
                    data_['z_shifted'] = z
                    d_s_x = data_.x_shifted
                    d_s_y = data_.y_shifted
                    d_s_z = data_.z_shifted

                if np.all([np.ptp(d_s_x)>=10.0, np.ptp(d_s_y)>=10.0, np.ptp(d_s_z)>=10.0]):

                    #########################
                    # save to dataframe
                    #########################
                    df_filtered = df_filtered.append(data_, ignore_index = True)

        aq += 1




################################
################################
# Bayesian inference step
################################
################################

# code from Metzner et al.
############################
# compute likelihood on parameter grid
def compLike(vp,v):
    return np.exp(-((v[0] - qGrid*vp[0])**2 + (v[1] - qGrid*vp[1])**2 + (v[2] - qGrid*vp[2])**2)/(2*a2Grid) - (3/2)*np.log(2*np.pi*a2Grid))

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


celltype_index = dict(zip(df_filtered.celltype.sort_values().unique(),
                np.arange(0,len(df_filtered.celltype.unique())).T))

for i, data in  df_filtered.groupby(['celltype']):

    # make DataFrame to save Parameter estimates
    df_Bayesian = pd.DataFrame()
    df_Bayesian2 = pd.DataFrame()

    activity_avg = []
    persistence_avg = []
    print(i)

    index = celltype_index[i]
    material = i

    vel = np.array([])
    count = 0

    meanPost1_all_P = []
    meanPost1_all_A = []
    meanPost1_length = []

    for j, data_ in data.groupby(['cell', 'trial', 'position', 'date', 'concentration']):

        data_ = data_[['x_shifted','y_shifted','z_shifted', 'frame']]
        data_.columns = ['x','y','z', 'frame']
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

            dz_ = df[df.frame==t+1].z.values - \
                    df[df.frame==t].z.values
            dz = np.append(dz,dz_)

            if (len(dx_) == 0):
                continue

        if len(dx)<15:
            continue

#         print(j)
        # Now lets begin Bayesian analysis of cell track
        veloSequ1 = np.zeros([len(dx), 3])
        veloSequ1[:,0] = dx/60.0 # um
        veloSequ1[:,1] = dy/60.0 # um
        veloSequ1[:,2] = dz/60.0 # um

        postSequ1 = compPostSequ(veloSequ1)
        meanPost1 = compPostMean(postSequ1)
        meanPost1_all_P = np.append(meanPost1_all_P, meanPost1[0])
        meanPost1_all_A = np.append(meanPost1_all_A, meanPost1[1])
        count += 1

        # append average persistence and activity values to dataframe
        data_Bayesian = {'cell':j[0],
                'celltype':i,
                'trial':j[1],
                'date':j[-2],
                'position':j[-3],
                 'concentration' : j[-1],
                'average_persistence':np.mean(meanPost1[0]),
                'activity ($\mu$m/sec)':np.mean(meanPost1[1]),
                'speed ($\mu$m/sec)': np.mean(np.sqrt(veloSequ1[:,0]**2 + \
                                    veloSequ1[:,1]**2 + veloSequ1[:,2]**2)),
                         'material':material}
        df_Bayesian = df_Bayesian.append(data_Bayesian, ignore_index=True)

        # append all inference values to dataframe
        for t in np.arange(len(veloSequ1)-1):#df.frame.unique():#np.arange(df.frame.min(), df.frame.max()):# + len(dx)):
            data_Bayesian2 = {'cell':j[0],
                              'celltype':i,
                              'frame':t,
                'Vx ($\mu$m/sec)':veloSequ1[:,0][int(t)],
                'Vy ($\mu$m/sec)':veloSequ1[:,1][int(t)],
                'Vz ($\mu$m/sec)':veloSequ1[:,2][int(t)],
                'speed ($\mu$m/sec)': np.sqrt(veloSequ1[:,0][int(t)]**2 + \
                                    veloSequ1[:,1][int(t)]**2 + veloSequ1[:,2][int(t)]**2),
                'trial':j[1],
                'date':j[-2],
                'position':j[-3],
                 'concentration' : j[-1],
                'material':material,
                'persistence':meanPost1[0][int(t)],
                'activity ($\mu$m/sec)':meanPost1[1][int(t)]}
            df_Bayesian2 = df_Bayesian2.append(data_Bayesian2, ignore_index=True)

    fname_ = '_'.join([i, '_all.csv'])
    df_Bayesian.to_csv('../../../data/processed_tracking_bayesian/20220920_3D_filtered_avg_'+fname_)
    df_Bayesian2.to_csv('../../../data/processed_tracking_bayesian/20220920_3D_filtered_'+fname_)
