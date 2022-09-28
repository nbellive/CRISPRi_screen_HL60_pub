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
# ip = 200.0/303.0 #(200 um per 303 pixels)
ip = 500.0/772.0 #(500 um per 772 pixels)

print('running')
######################################
## Run through all image files for EFon
######################################
files= glob.glob('../../../../../../../Volumes/homes//nbelliveau/Microscopy_RAW/iSIM/xIon/20220214/20220214_SC575_sgLAMTOR1_Heochst_Epi_BF_60secFR_3umZ_2/*t.ome.tif')
# files= glob.glob('../../../data/microscopy/20220208_KD_FNIbidi/20220208_KW-SC575_sgCtrl1_5dDMSO_FNcoat_30secFR_Heochst_phase_1/*Pos0.ome.tif')
print(files)

celltype = 'HL-60KW_SC575_sgLAMTOR1'
acqtime = '60sec'
threshold = 150

analysis_date = 20220223
framerate = 'NA'
date = 20220214
# celltype = 'HL-60KW_SC575_sgControl1'
obj = '20xAir'
trial = 2
media = 'L-15_10%FBS'
misc = '3D'
scope = 'iSIM_xIon'
ecm = 'collagen'
conc = '0.75mgml'
step = 3 #um
step = float(step)

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
        struct_img = _img.data[:,structure_channel,:,:,:]
        num_frames = len(struct_img[:,0,0,0])

        # dataframe to hold all the data
        df = pd.DataFrame()
        for t in range(num_frames):
            # img = struct_img[np.newaxis,:,:]
            img = struct_img[t, :, :,:]
            df = df.append(utils.segment_cells_img(t, img, threshold, ip, step),
                           ignore_index = True)
        df_track_temp = utils.tracking_track(df)
        df_track_temp['celltype'] = celltype
        df_track_temp['position'] = i
        df_track_temp['acqtime'] = acqtime

        df_track = df_track.append(df_track_temp, ignore_index=True)
#
df_track.to_csv('../../../data/processed_tracking/20220214_KDlines_3D/20220214_sgLAMTOR1_2_tracks.csv')

df_track_lab =  utils.tracking_label_simple(df_track, analysis_date, acqtime, date, celltype,
        ecm, conc, scope, obj, trial, media, misc)


# save to file
df_track_lab.to_csv('../../../data/processed_tracking/20220214_KDlines_3D/20220214_sgLAMTOR1_2_tracks.csv')
