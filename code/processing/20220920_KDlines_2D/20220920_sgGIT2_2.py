import sys
sys.path.insert(1, '../../../../motility_utils/')
import utils_2 as utils
import os
import glob
import aicsimageio
import numpy as np
import pandas as pd

# fluorescence image channel in file is 0
structure_channel = 0
# pixel conversion
ip = 200.0/303.0 #(200 um per 303 pixels)
print('running')
######################################
## Run through all image files
######################################
files= glob.glob('../../../../../../../Volumes/ExpansionHomesA//nbelliveau/Microscopy_RAW/Rio/20220920_kdLines/20220920_20xair_KWSC575_sgGIT2_dHL-60_FNcoated_30sec_2/*Pos0.ome.tif')
print(files)

structure_channel = 0
celltype = 'HL-60KW_SC575_sgGIT2'
acqtime = '30sec'
threshold = 200

analysis_date = 20220921
framerate = 'NA'
date = 20220920
# celltype = 'HL-60KW_SC575_sgControl1'
ecm = 'NA'
conc = 'NA'
obj = '20xAir'
scope = 'Rio_NikonTi'
trial = 2
media = 'L-15_10%FBS'
misc = '10ug/ml fibronectin'

# we're going to make a Pandas DataFrame to save all the track info
df_track = pd.DataFrame()

# loop through all files and perform segmentation/tracking
for f in files:
    print(f)
    _img = aicsimageio.readers.tiff_reader.TiffReader(f)
    for i, s in enumerate(_img.scenes):
        print('position ',i)
        _img.set_scene(s)
        # print(_img.shape)
        # break
        struct_img = _img.data[:,structure_channel,:,:]
        num_frames = len(struct_img[:,0,0])

        # dataframe to hold all the data
        df = pd.DataFrame()
        for t in range(num_frames):
            # img = struct_img[np.newaxis,:,:]
            img = struct_img[t, :,:]
            df = df.append(utils.segment_cells_2d(t, img, ip, threshold),
                           ignore_index = True)
        df_track_temp = utils.tracking_track_2d(df)
        df_track_temp['celltype'] = celltype
        df_track_temp['position'] = i
        df_track_temp['acqtime'] = acqtime

        df_track = df_track.append(df_track_temp, ignore_index=True)
#

df_track_lab =  utils.tracking_label_simple(df_track, analysis_date, acqtime, date, celltype,
        ecm, conc, scope, obj, trial, media, misc)


# save to file
df_track_lab.to_csv('../../../data/processed_tracking/20220920_KDlines_2D/20220920_sgGIT2_2_tracks.csv')
