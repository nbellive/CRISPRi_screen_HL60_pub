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
files= glob.glob('../../../../../../../Volumes/homes//nbelliveau/Microscopy_RAW/iSIM/xIon/20220315/20220313_KWSC575_sgITGA1_20x_60sFR_Heochst_BF_1.5mgmlCollagen_1/*0.ome.tif')
print(files)


celltype = 'HL-60KW_SC575_sgITGA1'
acqtime = '60sec'
threshold = 800#650

analysis_date = 20220318
framerate = 'NA'
date = 20220315
# celltype = 'HL-60KW_SC575_sgControl1'
obj = '20xAir'
trial = 1
media = 'L-15_10%FBS'
misc = '3D'
scope = 'iSIM_xIon'
ecm = 'collagen'
conc = '1.5mgml'
step = 3.0 #um
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
        print(_img.shape)
        # break
        struct_img = _img.data[:,structure_channel,:,:,:]
        num_frames = len(struct_img[:,0,0,0])
        print(num_frames)
        # dataframe to hold all the data
        df = pd.DataFrame()
        for t in range(num_frames):
            # img = struct_img[np.newaxis,:,:]
            if t>=58:
                continue
            else:
                img = struct_img[t, :, :,:]
                df = df.append(utils.segment_cells_img(t, img, threshold, ip, step),
                               ignore_index = True)
        df_track_temp = utils.tracking_track(df)
        df_track_temp['celltype'] = celltype
        df_track_temp['position'] = i
        df_track_temp['acqtime'] = acqtime

        df_track = df_track.append(df_track_temp, ignore_index=True)
#
df_track.to_csv('../../../data/processed_tracking/20220315_KDlines_3D/20220315_sgITGA1_1_5_1_tracks.csv')

df_track_lab =  utils.tracking_label_simple(df_track, analysis_date, acqtime, date, celltype,
        ecm, conc, scope, obj, trial, media, misc)


# save to file
df_track_lab.to_csv('../../../data/processed_tracking/20220315_KDlines_3D/20220315_sgITGA1_1_5_1_tracks.csv')
