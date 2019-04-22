# -*- coding: utf-8 -*-

# load
import supereeg as se
import numpy as np
import glob

def make_slices(nifti, gif_path, time_index=range(100, 200), slice_index=range(-4,52,7), name=None, vmax=2, duration=1000, symmetric_cbar=True, **kwargs):
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

    # get Haxby mask
    # mask = datasets.fetch_haxby().mask
    images = []
    num_slice = len(slice_index)
    nrow = np.floor(np.sqrt(num_slice))
    ncol = np.ceil(num_slice / nrow)

    for i in time_index:
        temp_img = []
        for loc in slice_index:
            nii_i = image.index_img(nifti, i)
            if loc == slice_index[len(slice_index) - 1]:
                display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=True, symmetric_cbar=symmetric_cbar, vmax=vmax, **kwargs)
            else:
                display = ni_plt.plot_stat_map(nii_i, cut_coords=[loc], colorbar=False, symmetric_cbar=symmetric_cbar, vmax=vmax, **kwargs)
            # display.add_contours(mask, levels=[.5], filled=True, colors='y')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            temp_img.append(buf)
            plt.close()
            #buf.close()
        first = Image.open(temp_img[0])
        width, height = first.size
        last = Image.open(temp_img[len(temp_img) - 1])
        lw, lh = last.size
        frame = Image.new('RGB', (int((ncol - 1) * width + lw), int(nrow * height)))
        crow = 0
        ccol = 0
        print(i)
        for buf in temp_img:
            buf.seek(0)
            frame.paste(im=Image.open(buf), box=(int(ccol * width), int(crow * height)))
            crow += int(np.floor(ccol / (ncol - 1)))
            ccol = (ccol + 1) % ncol
        images.append(frame)

    if name is None:
        gif_outfile = os.path.join(gif_path, 'gif_' + str(min(time_index)) + '_' + str(max(time_index)) + '.gif')
    else:
        gif_outfile = os.path.join(gif_path, str(name) + '.gif')

    # creates the gif from the frames
    images[0].save(gif_outfile, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)


fnames = glob.glob('*_recon.bo')

for i, fname in enumerate(fnames):
    bo = se.load(fname)
    nii = bo.to_nii(template='std', vox_size=6)

    # time_index = np.arange(200*180, len(bo.data[0]) - 200*420)
    time_index = np.arange(0,6000)[::2]
    #C:\Users\tmunt\Documents\gif   \\dartfs\\rc\\lab\\D\\DBIC\\CDL\\f003f64\\gifs

    #change duration!
    make_slices(nii, '\\dartfs\\rc\\lab\\D\\DBIC\\CDL\\f003f64\\gifs', time_index=time_index, slice_index=range(-50,50, 4), name=fname.split('.')[0] + '.gif', vmax=np.amax(bo.data[0]), symmetric_cbar=False, duration=10, alpha=0.4, display_mode='y')