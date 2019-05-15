# -*- coding: utf-8 -*-

# load
import supereeg as se
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
from nilearn import plotting as ni_plt
from nilearn import _utils
from nilearn import image
import io
import os
import sys
import cProfile, pstats
import multiprocessing as mp
import dill

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

def wrapper(func, **kwargs):
    def wrapped(*args):
        return func(*args, **kwargs)
    return wrapped

def helper(time_index, nifti=None, path=None, slice_index=range(-4,52,7), vmax=3.5, symmetric_cbar=False, alpha=0.5, **kwargs):
    # needs time_index, nifti, and path

    num_slice = len(slice_index)
    nrow = int(np.floor(np.sqrt(num_slice)))
    ncol = int(np.ceil(num_slice / nrow))

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, 10))  # move out of for loop, set ax
    ax = ax.reshape(-1)

    displays = []
    # nifti = _utils.check_niimg_4d(nifti)
    for currax, loc in enumerate(slice_index):
        nii_i = image.index_img(nifti, time_index[0])
        if loc == slice_index[len(slice_index) - 1]:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        else:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=False, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        displays.append(display)
    out = os.path.join(path,str(fname) + str(time_index[0])+'.png')
    f = open(out, 'w+')
    f.close()
    plt.savefig(out, format='png')

    for i in time_index[1:]:
        nii_i = image.index_img(nifti, i)
        for dindex, display in enumerate(displays):
            if dindex == len(displays) - 1:
                display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar,
                                               figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
            else:
                display.add_overlay(nii_i, colorbar=False, vmax=vmax, alpha=alpha)
        out = os.path.join(path, str(fname) + str(i)) + '.png'
        f = open(out, 'w+')
        f.close()
        plt.savefig(out, format='png')

    plt.close()
    return os.getpid()

def helper2(time_index):
    # needs time_index, nifti, and path
    nifti = None
    path = None
    slice_index = range(-4, 52, 7)
    vmax = 3.5
    symmetric_cbar = False
    alpha = 0.5
    fname = 'test.bo'
    bo = se.load(fname)
    nii = bo.to_nii(template='std', vox_size=6)

    num_slice = len(slice_index)
    nrow = int(np.floor(np.sqrt(num_slice)))
    ncol = int(np.ceil(num_slice / nrow))

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, 10))  # move out of for loop, set ax
    ax = ax.reshape(-1)

    displays = []
    # nifti = _utils.check_niimg_4d(nifti)
    for currax, loc in enumerate(slice_index):
        nii_i = image.index_img(nifti, time_index[0])
        if loc == slice_index[len(slice_index) - 1]:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        else:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=False, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        displays.append(display)

    plt.savefig(os.path.join(path, str(time_index[0])+'.png'), format='png')

    for i in time_index[1:]:
        nii_i = image.index_img(nifti, i)
        for display in displays:
            display.add_overlay(nii_i, colorbar=False, vmax=vmax, alpha=alpha)
        plt.savefig(os.path.join(path, str(i)) + '.png', format='png')

    plt.close()
    return os.getpid()



if __name__ == "__main__":
    nworkers = mp.cpu_count()
    fname = sys.argv[1]
    try:
        vmax = int(sys.argv[2])
    except:
        vmax = 3.5
    bo = se.load(fname)
    timepoints = bo.data.shape[0]
    ranges = np.array_split(np.arange(timepoints), nworkers) #nworkers
    nii = bo.to_nii2(template='std', vox_size=6)
    path = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/gifs' # 'C:\\Users\\tmunt\\Documents\\gif'
    helpwrap = wrapper(helper, nifti=nii, path=path, slice_index=range(-50, 50, 4), vmax=vmax, symmetric_cbar=True, display_mode='y')
    pr = cProfile.Profile()
    pr.enable()
    async_results = []
    # pool = mp.Pool()
    kw = {'nifti': nii, 'path': path, 'slice_index': range(-50, 50, 4), 'vmax': vmax, 'symmetric_cbar': True,
              'display_mode': 'y'}

    processes = []
    for range in ranges:
        processes.append(mp.Process(target=helper, args=(range,), kwargs=kw))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    gif_outfile = os.path.join(path, fname.split('.')[0] + '.gif')

    image_fnames = glob.glob(os.path.join(path, fname + '*.png'))
    images = []

    for im_fname in image_fnames:
        images.append(Image.open(im_fname))
    # creates the gif from the frames
    images[0].save(gif_outfile, format='GIF', append_images=images[1:], save_all=True, duration=200, loop=0)
    pr.disable()
    s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())