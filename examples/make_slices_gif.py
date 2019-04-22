# -*- coding: utf-8 -*-

# load
import supereeg as se
import numpy as np
import glob

fnames = glob.glob('*_recon.bo')

for i, fname in enumerate(fnames):
    bo = se.load(fname)
    nii = bo.to_nii(template='std', vox_size=6)

    # time_index = np.arange(200*180, len(bo.data[0]) - 200*420)
    time_index = np.arange(0,6000)[::2]
    #C:\Users\tmunt\Documents\gif   \\dartfs\\rc\\lab\\D\\DBIC\\CDL\\f003f64\\gifs

    #change duration!
    nii.make_sliced_gif('\\dartfs\\rc\\lab\\D\\DBIC\\CDL\\f003f64\\gifs', time_index=time_index, slice_index=range(-50,50, 4), name=fname.split('.')[0] + '.gif', vmax=np.amax(bo.data), symmetric_cbar=False, duration=10, alpha=0.4, display_mode='y')
