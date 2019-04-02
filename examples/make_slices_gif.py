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

# Code source: Lucy Owen & Andrew Heusser, modified by Tudor Muntianu
# License: MIT

# load
import supereeg as se
import numpy as np

# load example data
# bo = se.load('example_data')

# simulate 100 locations
locs = se.simulate_locations(n_elecs=100)

# simulate brain object
bo = se.simulate_bo(n_samples=400, sample_rate=100, cov='random', locs=locs, noise =.1)

# convert to nifti
nii = bo.to_nii(template='std', vox_size=6)

# make gif
# '/your/path/to/gif/'
nii.make_sliced_gif('C:\\Users\\tmunt\\Documents\\gif', time_index=np.arange(10), slice_index=range(-4,52,4), name='sample_gif', alpha=0.7)
