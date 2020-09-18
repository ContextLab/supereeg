# -*- coding: utf-8 -*-
"""
=============================
Make gif
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.  We then convert the reconstruction to a nifti and plot 3 consecutive timepoints
first with the plot_glass_brain and then create .png files and compile as a gif.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# load
import supereeg as se
import numpy as np

# load example data
bo = se.load('example_data')

# convert to nifti
nifti = bo.to_nii(template='gray', vox_size=6)

# make gif, default time window is 0 to 10, but you can specifiy by setting a range with index
# nifti.make_gif('C:\\Users\\tmunt\\Documents\\gif', index=np.arange(5), name='sample_gif')

nifti.make_sliced_gif('C:\\Users\\tmunt\\Documents\\gif', time_index=np.arange(3), name='sample_gif')
