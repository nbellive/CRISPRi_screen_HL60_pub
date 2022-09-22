
import glob
import os

import numpy as np
import pandas as pd
import scipy.signal

import skimage

# A whole bunch of skimage stuff
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.restoration
import skimage.segmentation
import skimage.transform
from skimage.feature import register_translation

import multiprocessing
# following https://sedeh.github.io/python-pandas-multiprocessing-workaround.html
from joblib import Parallel, parallel_backend, delayed

import sys
sys.path.append('../../../../')
import bebi103



def bebi103_thresh(im, selem, white_true=True, k_range=(0.5, 1.5),
                   min_size=100):
    """
    Threshold image.  Morphological mean filter is
    applied using selem.
    """
    # Determine comparison operator
    if white_true:
        compare = np.greater
        sign = -1
    else:
        compare = np.less
        sign = 1

    # Do the mean filter
    im_mean = skimage.filters.rank.mean(im, selem)

    # Compute number of pixels in binary image as a function of k
    k = np.linspace(k_range[0], k_range[1], 100)
    n_pix = np.empty_like(k)
    for i in range(len(k)):
        n_pix[i] = compare(im, k[i] * im_mean).sum()

    # Compute rough second derivative
    dn_pix_dk2 = np.diff(np.diff(n_pix))

    # Find index of maximal second derivative
    max_ind = np.argmax(sign * dn_pix_dk2)

    # Use this index to set k
    k_opt = k[max_ind - sign * 2]

    # Threshold with this k
    im_bw = compare(im, k_opt * im_mean)

    # Remove all the small objects
    im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=min_size)

    return im_bw, k_opt


def highpass_filt(im):
    im_fft = np.fft.fft2(im)
    fshift = np.fft.fftshift(im_fft)

    ncols, nrows = 1024, 1024

    # Build and apply a Gaussian filter.
    sigmax, sigmay = 50, 50
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
#     gmask2 = np.sqrt((X - cx)**2 + (Y - cy)**2)

    ftimagep = fshift * (-1*gmask + 1)

    # Finally, take the inverse transform and show the blurred image
    return np.fft.ifft2(ftimagep)

def segment_cells(t):
    print(t)
    df = pd.DataFrame()
    # now loop over all t
    # for t in np.arange(0,60):
    count = 0


    shift = [0.0, 0.0]

    im_labeled_temp = np.zeros((1024,1024))
    # df_tminus = df[df.frame==(t-1)].sort_values('cell').copy()
    disp_arr = []

    # for z in np.arange(0,z_size):
    for z in np.arange(0,num_z):
        # print(fname, ' : ', t, ' z: ', z)
        im_fname_405 = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_405_t' + str(t+1).zfill(3)  + '_z'+ str(z+1).zfill(3) + '.tif'
        im_temp = skimage.io.imread(im_fname_405)
        im_temp[im_temp <= threshold_405] = 0

        # Make the structuring element 50 pixel radius disk
        selem = skimage.morphology.disk(50)

        # Threshhold based on mean filter
        im_bw, k = bebi103_thresh(im_temp, selem, white_true=True, min_size=50)
        # Label binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(
                                  im_bw, background=0, return_num=True)

        # Get properties
        im_props = skimage.measure.regionprops(im_labeled)

        for i, prop in enumerate(im_props):
            if im_labeled_temp[int(prop.centroid[0]),int(prop.centroid[1])] == 0:
                x = (prop.centroid[1])*ip
                y = (prop.centroid[0])*ip

                # find z by a weighted average of signal intensity across stacks and area that likely has entire nuclei
                zframes = z_size - z
                z_sum = np.zeros(13)
                for k in range(0,np.min([zframes,13])):
                    im_fname_405_ = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_405_t' + str(t+1).zfill(3)  + '_z'+ str(z+k+1).zfill(3) + '.tif'
                    im = np.zeros([1024+30,1024+30])
                    im[15:-15, 15:-15] = skimage.io.imread(im_fname_405_)
                    im[im <= threshold_405] = 0
                    z_sum[k] = im[(15+int(prop.centroid[0])-15):(15+int(prop.centroid[0])+15), (15+int(prop.centroid[1])-15):(15+int(prop.centroid[1])+15)].sum().sum()
                z_max = (np.arange(13)*z_sum).sum()/z_sum.sum()
                if np.min([zframes,13]) == 1:
                        z_max = 1.0
                z_pos = step*(float(z) + z_max)

                # append data to df
                data = {'cell':count, 'frame':t, 'framerate':framerate, 'x':x, 'y':y, 'z':z_pos,
                        'date':date, 'celltype':celltype, 'material':ecm,
                        'concentration':conc, 'scope':scope, 'magnification':obj,
                        'trial':trial, 'x_corr':shift[1], 'y_corr':shift[0], 'fMLP':fmlp, 'E_field':efield}
                df = df.append(data, ignore_index=True)

                count += 1
        # make temp make to use for comparing identified objects in next time point
        im_labeled_temp = im_labeled.copy()
    # print(t)
    # df.to_csv(fname_pre+'_'+str(t)+'.csv')

    return df.values.tolist()
        # print(df[df.cell==10])

def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)

if __name__ == "__main__":
    # multiprocessing.set_start_method('forkserver')
    # Sydney 20xwater (200 um/ 303 pixels)
    ip = 200.0/303.0

    z_size = 17

    results = []

    # threshold_405 = 1000

    filelist_fl = ['/Volumes/Belliveau_RAW_1_JTgroup/Rio/20190116/20190116_dHL60_20x_ibidi_1mgml_collagen_50Vsaltbridge_25V_3/20190116_Rio_20xair_66_5_dHL60_collagen_1.0mgml_3_405.tif']


    # Grab image session details
    fname = os.path.basename(filelist_fl[0])
    date, scope, obj, framerate, step, celltype, ecm, conc, trial, _ = fname.split('_')
    step = float(step)

    fmlp = 0.0
    efield = 5.0

    # check number of frames
    num_frames = len(glob.glob(os.path.dirname(filelist_fl[0]) + '/images/*405_*_z001.tif'))
    num_z = len(glob.glob(os.path.dirname(filelist_fl[0]) + '/images/*405_t001_*.tif'))

    # determine the threshold backgroun noise
    # t = 0
    # z = 0
    fname_pre = "_".join([date, scope, obj, framerate, str(int(step)), celltype, ecm, conc, trial])
    # im_fname_405 = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_405_t' + str(t+1).zfill(3)  + '_z'+ str(z+1).zfill(3) + '.tif'
    #
    # im_temp = skimage.io.imread(im_fname_405)
    # im_temp[im_temp <= 800] = 0
    # # Make the structuring element 50 pixel radius disk
    # selem = skimage.morphology.disk(50)
    #
    # # Threshhold based on mean filter
    # im_bw, k = bebi103_thresh(im_temp, selem, white_true=True, min_size=50)
    #             # Label binary image; backward kwarg says value in im_bw to consider backgr.
    # im_labeled, n_labels = skimage.measure.label(
    #                           im_bw, background=0, return_num=True)
    # im_labeled[im_labeled == 0] = -1
    # im_labeled[im_labeled >= 1] = 0
    # im_labeled[im_labeled == -1] = 1
    # im_temp_ = skimage.io.imread(im_fname_405)
    # im_temp_ = im_labeled*im_temp_
    # im_temp_ = np.ravel(im_temp_)
    # threshold_405 = np.average(im_temp_[im_temp_!=0])
    # print('average background value: ',threshold_405)
    threshold_405 = 2000.0

    im_fname_pre = "_".join([date, scope, obj, framerate, str(int(step)), celltype, ecm, conc, trial])

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    for t in range(num_frames):
        pool.apply_async(segment_cells, args=(t,  ), callback=collect_results)
    pool.close()
    pool.join()

    # Converts list of lists to a data frame
    df = pd.DataFrame(results)
    df.columns = ['E_field','cell','celltype','concentration','date','fMLP','frame','framerate','magnification','material','scope','trial','x','x_corr','y','y_corr','z']
    #
    print('List processing complete.')
    df.to_csv(fname_pre + '_cellpositions_multi.csv')



####################################
## okay, now lets write code to identify cells.

df_track = df[df.frame==0].sort_values(by=['cell'])
# num_cells = len(df_track)

for t in np.arange(1,num_frames):
    disp_arr = []
    print(t)
    num_cells = len(df[df.frame==t-1].cell.unique())
    for cell in df[df.frame==t].cell.unique():
        disp_arr_temp = np.zeros(3+num_cells)
        # print(df[(df.cell==cell) & (df.frame==t)].x)
        disp_arr_temp[0] = df[(df.cell==cell) & (df.frame==t)].x
        disp_arr_temp[1] = df[(df.cell==cell) & (df.frame==t)].y
        disp_arr_temp[2] = df[(df.cell==cell) & (df.frame==t)].z

        df_tminus = df[df.frame==t-1]
        # print(df_tminus['x'])
        # print(disp_arr_temp[0])
        disp_arr_temp[3:] = np.sqrt((df_tminus['x'] - disp_arr_temp[0])**2 + \
                       (df_tminus['y'] - disp_arr_temp[1])**2 + \
                       (df_tminus['z'] -  disp_arr_temp[2])**2).values

        disp_arr = np.append(disp_arr,disp_arr_temp)

    # reshape array to correct size (columns: i, x, y, number of cells considered; rows: number of items considered)
    disp_arr = disp_arr.reshape(int(len(disp_arr)/(3+len(df_tminus))),3+len(df_tminus))
#     print(disp_arr)
    # note that I should sort such that I assign closest objects first! Lets try.
    disp_arr_sorted = np.min(disp_arr[:,3:].copy(),axis=0)
    disp_arr_sorted_ind = np.argsort(disp_arr_sorted)


    for cell in disp_arr_sorted_ind:
        # look for an objects that are close to each other between this and prior time point
#         print(disp_arr[:,4+cell].min())
        if  disp_arr[:,3+cell].min() <= 30.0:
            disp_ind = np.where(disp_arr[:,3+cell] == disp_arr[:,3+cell].min())[0][0]

            # here I could consider checking the intensity values for +/- a couple z values
            # in actual image and pick z with highest intensity value.
            x_pos = disp_arr[disp_ind,0]
            y_pos = disp_arr[disp_ind,1]
            z_pos = disp_arr[disp_ind,2]


            data = df[(df.frame==t) & (df.cell==disp_ind)]
            data.replace({'cell': disp_ind}, cell)
            df_track = df_track.append(data, ignore_index=True)
            # print('cell matched: ', cell)
            # print(df[df.cell==cell])
            # 'remove' object/cell that has been assigned from the current
            # array of objects, by making it infinitely far away
            disp_arr[disp_ind,4:] = np.inf

    # for any cell from the previous time point which wasn't assigned, assume
    # it was lost (i.e. went out of frame)
    for cell in np.arange(0,num_cells):
        if cell not in df[df.frame==t].cell.unique():
            data = df[(df.frame==t) & (df.cell==cell)]
            data.replace({'cell': cell, 'x':data.x, 'y':data.y, 'z':data.z}, {'cell': cell, 'x':np.inf, 'y':np.inf, 'z':np.inf})
            df_track = df_track.append(data, ignore_index=True)

    for prop_ind in np.arange(0,len(disp_arr[:,0])):
        count = 0
        if 30.01 <= disp_arr[prop_ind,3:].min() <= 10000.0:
            x_pos = disp_arr[prop_ind,0]
            y_pos = disp_arr[prop_ind,1]
            z_pos = disp_arr[prop_ind,2]

            data = df[(df.frame==t) & (df.cell==prop_ind)]
            data.replace({'cell': prop_ind}, num_cells + count)
            df_track = df_track.append(data, ignore_index=True)
            count += 1

#####################################
# perform autocorrelation analysis using brightfield images

df_xy  = pd.DataFrame()

for t in np.arange(0,num_frames):
    count = 0
    print(fname, ' : ', t)
#     if t >= 6:
#         break

    # calculate any drift in x,y
    # subpixel precision
    if t != 0:
        im_fname_bf_t0 = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_phase_t' + str(t).zfill(3)  + '_z'+ str(1).zfill(3) + '.tif'
        im_bf_t0 = skimage.io.imread(im_fname_bf_t0)
        im_fname_bf_t1 = os.path.dirname(filelist_fl[0]) + '/images/' + fname_pre + '_phase_t' + str(t+1).zfill(3)  + '_z'+ str(1).zfill(3) + '.tif'
        im_bf_t1 = skimage.io.imread(im_fname_bf_t1)

        shift, error, diffphase = register_translation(highpass_filt(im_bf_t0),highpass_filt(im_bf_t1), 100)
        # drift_corr_y += shift[0]
        # drift_corr_x += shift[1]
    else:
        shift = [0.0, 0.0]

    for cell in df[df.frame==t].cell.unique():

        data = {'cell':cell, 'frame':t, 'x_corr':shift[1], 'y_corr':shift[0]}
        df_xy = df_xy.append(data, ignore_index=True)

# append the xy corrections to the main DataFrame
df_track = pd.merge(df_track, df_xy, on=['cell','frame'])

#####################################
# Save to file

df_track.to_csv(fname_pre + '_celltracking_multi.csv')
