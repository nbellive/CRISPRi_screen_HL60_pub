
import glob
import os

import numpy as np
import pandas as pd
# import scipy.signal
#
import skimage
#
# A whole bunch of skimage stuff
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.restoration
import skimage.segmentation
import skimage.transform
import skimage.exposure
from skimage.feature import register_translation

import sys

################
# steps
# 1. load in the cell tracking data
# 2. go through each image assocated with cells and save cropped area (50x50?)
#    Append this to an array that'll hold 10 images in 1 row.
# 3. Save array as jpeg
# 4. convert image series of jpeg into .avi
################

############
# Need to also identify cells that 'move' and are worth looking at
############

ip = 200.0/303.0

df = pd.read_csv('20190116_Rio_20xair_66_5_dHL60_collagen_1.0mgml_0_celltracking_multi_good.csv')

filelist_fl = ['/Volumes/Belliveau_RAW_1_JTgroup/Rio/20190116/20190116_dHL60_20x_ibidi_1mgml_collagen_50Vsaltbridge_25V_0/20190116_Rio_20xair_66_5_dHL60_collagen_1.0mgml_0_405.tif']

# Grab image session details
fname = os.path.basename(filelist_fl[0])
date, scope, obj, framerate, step, celltype, ecm, conc, trial, _ = fname.split('_')
# obj = '20xwater'
step = float(step)

# finds cells that move a lot
motile_cells = []
df_cell_groups = df.groupby('cell')
for cell, data in df_cell_groups:
    # print(np.ptp(data.x))
    # print(len(data.x.unique()))
    if np.ptp(data.x) >=50.0:
        if (data.frame.min() == 0) and  (len(data.x.unique())) >= 20:
            motile_cells = np.append(motile_cells, cell)
print(motile_cells)
fname_pre = "_".join([date, scope, obj, framerate, str(int(step)), celltype, ecm, conc, trial])

df_cells = df.sort_values(by=['frame']).groupby('frame')
for t, data in df_cells:
    if t >= 20:
        continue
    print(np.min([len(motile_cells), 10]))
    im_arr = np.zeros([60, 60*np.min([len(motile_cells), 10])])
    # print(df.cell.min())
    for i, cell in enumerate(motile_cells):
        if cell == 28:
            continue
        if i >= np.min([len(motile_cells), 10]):
            continue
        data_ = data[data.cell == cell]
        x = np.around(data_.x.values/ip)
        y = np.around(data_.y.values/ip)
        print(x,y)
        z = np.around(data_.z.values/5) +1
        if z >= 17:
            z = 16
        # print(x,y,z)
        # if np.any([x,y,z]):
        #     continue
        im_fname = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_phase_t' + str(int(t)+1).zfill(3)  + '_z'+ str(int(z)+1).zfill(3) + '.tif'
        im = skimage.io.imread(im_fname)
        x_ = int(x)
        y_ = int(y)
        im_arr[:, 60*i:(60*i + 60)] = im[(y_-30):(y_+30), (x_-30):(x_+30)]

    im_arr = skimage.exposure.rescale_intensity(im_arr, out_range=(0,1))
    im_fname_save = 'images/' + fname_pre + '_brightfield_t' + str(int(t)+1).zfill(3) + '_cells.jpg'
    skimage.io.imsave(im_fname_save, im_arr, quality=100)
#

# df_cells = df.groupby('cell')
# for cell, data in df_cells:
#     data = data.sort_values(by=['frame'])
#     x = np.around(data.x.values/ip)
#     y = np.around(data.y.values/ip)
#     z = np.around(data.z.values/3)-2
#
#
#     for i, t in enumerate(data.frame):
#         # print(int(t), int(z[i]))
#         im_fname = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_brightfield_t' + str(int(t)+1).zfill(3)  + '_z'+ str(int(z[i])+1).zfill(3) + '.tif'
#         im = skimage.io.imread(im_fname)
#         x_ = int(x[i])
#         y_ = int(y[i])
#         im = im[(y_-30):(y_+30), (x_-30):(x_+30)]
#
#         im_fname_save = fname_pre + '_brightfield_t' + str(int(t)+1).zfill(3)  + '_z'+ str(int(z[i])+1).zfill(3) + str(cell) + '.jpg'
#         skimage.io.imsave(im_fname_save, im*2, quality=100)
#     break


import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 3, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
