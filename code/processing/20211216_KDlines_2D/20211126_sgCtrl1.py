import sys
sys.path.insert(1, '../../../../motility_utils/')
import utils#as utils
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
## Run through all image files for EFon
######################################
files= glob.glob('../../../../../../../Volumes/homes//nbelliveau/Microscopy_RAW/Rio/20211216_KDlines_5dayDMSO/20211126_sgCtrl1_5dDMSO_ibidi_30sec_1/*Pos0.ome.tif')
print(files)

structure_channel = 0
celltype = 'sgCtrl1'
aqcTime = '30sec'
threshold = 600

# we're going to make a Pandas DataFrame to save all the track info
df_track = pd.DataFrame()

# loop through all files and perform segmentation/tracking
for f in files:
    print(f)
    _img = aicsimageio.readers.tiff_reader.TiffReader(f)
    for i, s in enumerate(_img.scenes):
        print('position ',i)
        _img.set_scene(s)

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
        df_track_temp['aqctime'] = aqcTime

        df_track = df_track.append(df_track_temp, ignore_index=True)
#
# save to file
df_track.to_csv('20211126_sgCtrl1_tracks.csv')
