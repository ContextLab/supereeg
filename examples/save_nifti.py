# -*- coding: utf-8 -*-
"""
=============================
Predict unknown location
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se
import seaborn as sns
import numpy as np
from nilearn import plotting

# load example data
bo = se.load('example_data')

bo.data = bo.data.loc[:1000,:]
bo.sessions = bo.sessions.loc[:1000]

# load example model
model = se.load('example_model')

# fill in the missing timeseries data
bo = model.predict(bo)

# save as nifti
data = bo.to_nifti('test.nii')

# Import image processing tool
from nilearn import image

# Compute the voxel_wise mean of functional images across time.
# Basically reducing the functional image from 4D to 3D
mean_haxby_img = image.index_img(data, 0)

# Visualizing mean image (3D)
plotting.plot_glass_brain(mean_haxby_img, title='plot_glass_brain')
plotting.show()

# sns.heatmap(np.divide(bo.locs, new_locs))
# sns.plt.show()
