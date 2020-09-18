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
import cProfile, pstats

def sliced_gif_eff(nifti, gif_path, time_index=range(100, 200), slice_index=range(-4,52,7), name=None, vmax=3.5, duration=1000, symmetric_cbar=False, alpha=0.5, **kwargs):
    """
    Plots series of nifti timepoints as nilearn plot_anat_brain in .png format

    :param nifti: Nifti object to be plotted
    :param gif_path: Directory to save .png files
    :param time_index: Time indices to be plotted
    :param slice_index: Coordinates to be plotted
    :param name: Name of output gif
    :param vmax: scale of colorbar (IMPORTANT! change if scale changes are necessary)
    :param kwargs: Other args to be passed to nilearn's plot_anat_brain
    :return:
    """

    images = []
    num_slice = len(slice_index)
    nrow = int(np.floor(np.sqrt(num_slice)))
    ncol = int(np.ceil(num_slice / nrow))

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, 10))  # move out of for loop, set ax
    ax = ax.reshape(-1)

    displays = []
    nifti = _utils.check_niimg_4d(nifti)
    for currax, loc in enumerate(slice_index):
        nii_i = image.index_img(nifti, 0)
        if loc == slice_index[len(slice_index) - 1]:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        else:
            display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=False, symmetric_cbar=symmetric_cbar,
                                           figure=fig, axes=ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        displays.append(display)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images.append(Image.open(buf))

    for i in time_index[1:]:
        nii_i = image.index_img(nifti, i)
        for display in displays:
            display.add_overlay(nii_i, colorbar=False, vmax=vmax, alpha=alpha)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf))

    plt.close()

    if name is None:
        gif_outfile = os.path.join(gif_path, 'gif_' + str(min(time_index)) + '_' + str(max(time_index)) + '.gif')
    else:
        gif_outfile = os.path.join(gif_path, str(name) + '.gif')

    # creates the gif from the frames
    images[0].save(gif_outfile, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)


def sliced_gif(nifti, gif_path, time_index=range(100, 200), slice_index=range(-4,52,7), name=None, vmax=3.5, duration=1000, symmetric_cbar=False, alpha=0.5, **kwargs):
    """
    Plots series of nifti timepoints as nilearn plot_anat_brain in .png format

    :param nifti: Nifti object to be plotted
    :param gif_path: Directory to save .png files
    :param time_index: Time indices to be plotted
    :param slice_index: Coordinates to be plotted
    :param name: Name of output gif
    :param vmax: scale of colorbar (IMPORTANT! change if scale changes are necessary)
    :param kwargs: Other args to be passed to nilearn's plot_anat_brain
    :return:
    """

    images = []
    num_slice = len(slice_index)
    nrow = int(np.floor(np.sqrt(num_slice)))
    ncol = int(np.ceil(num_slice / nrow))

    for i in time_index:
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10,10)) # move out of for loop, set ax
        ax = ax.reshape(-1)
        for currax, loc in enumerate(slice_index):
            nii_i = image.index_img(nifti, i)
            if loc == slice_index[len(slice_index) - 1]:
                display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar, figure=fig, axes = ax[currax], vmax=vmax, alpha=alpha,  **kwargs)
            else:
                display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=False, symmetric_cbar=symmetric_cbar, figure=fig, axes = ax[currax], vmax=vmax, alpha=alpha, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close()

    if name is None:
        gif_outfile = os.path.join(gif_path, 'gif_' + str(min(time_index)) + '_' + str(max(time_index)) + '.gif')
    else:
        gif_outfile = os.path.join(gif_path, str(name) + '.gif')

    # creates the gif from the frames
    images[0].save(gif_outfile, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

# fnames = glob.glob('*_recon.bo')
fnames = ['test.bo']

for i, fname in enumerate(fnames):
    bo = se.load(fname)

    nii = bo.to_nii(template='std', vox_size=6)

    # time_index = np.arange(200*180, len(bo.data[0]) - 200*420)
    time_index = np.arange(0,9)
    #C:\\Users\\tmunt\\Documents\\gif   \\dartfs\\rc\\lab\\D\\DBIC\\CDL\\f003f64\\gifs

    sg_wr = wrapper(sliced_gif, nii, 'C:\\Users\\tmunt\\Documents\\gif', time_index=time_index, slice_index=range(-50,50, 4), name='test', vmax=3.5, symmetric_cbar=True, duration=200, alpha=0.6, display_mode='y')
    sge_wr = wrapper(sliced_gif_eff, nii, 'C:\\Users\\tmunt\\Documents\\gif', time_index=time_index, slice_index=range(-50,50, 4), name='testeff', vmax=3.5, symmetric_cbar=True, duration=200, alpha=0.6, display_mode='y')

    # print("regular: " + str(timeit.timeit(sg_wr, number=1)))
    # print("eff: " + str(timeit.timeit(sge_wr, number=1)))
    pr = cProfile.Profile()
    pr.enable()
    sliced_gif(nii, 'C:\\Users\\tmunt\\Documents\\gif', time_index=time_index, slice_index=range(-50,50, 4), name=fname.split('.')[0] + '.gif', vmax=3.5, symmetric_cbar=True, duration=200, alpha=0.6, display_mode='y')
    # sliced_gif_eff(nii, 'C:\\Users\\tmunt\\Documents\\gif', time_index=time_index, slice_index=range(-50,50, 4), name='testeff', vmax=3.5, symmetric_cbar=True, duration=200, alpha=0.6, display_mode='y')
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

