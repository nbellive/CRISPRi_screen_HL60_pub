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


# Function from https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
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

# path to data
files = glob.glob('../../../../../../../../Volumes/ExpansionHomesA/nbelliveau/microscopy_raw/2022_CollinsLab_HTChemotaxis-Nathan-TheriotLab-5dayDiff/mat files_updated/*/processed*.mat')

df = pd.DataFrame()
for f in files:
    vig=loadmat(f)
    df_temp = pd.DataFrame.from_dict(vig['mer1'])
    df_temp['date'] = f.split('/')[-2][:10]
    df = df.append(df_temp, ignore_index = True)

df.to_csv('../../../data/processed_tracking/20220404_chemotaxis/20220404_chemotaxis_compiled_data_Matlab_to_DataFrame.csv')
