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

# load example data
bo = se.load('example_data')

# load example model
model = se.load('example_model')

# fill in the missing timeseries data with original set of locations
reconstructed_bo = model.predict(bo, nearest_neighbor = False)

# plot locations
bo.plot_locs('/Users/lucyowen/Desktop/bo_no_nn.pdf')

# fill in the missing timeseries data
reconstructed_bo_nn = model.predict(bo, nearest_neighbor = True)

# plot new reconstructed locations using the nearest neighbor model location
bo.plot_locs('/Users/lucyowen/Desktop/bo_nn.pdf')

# print out info on new brain object
reconstructed_bo.info()

# save as nifti
reconstructed_nifti = reconstructed_bo.to_nii()




