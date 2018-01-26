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


# # plot locations
# bo.plot_locs()

# fill in the missing timeseries data with nearest neighbor locations
reconstructed_bo_nn = model.predict(bo, force_update=True)

reconstructed_nifti = reconstructed_bo_nn.to_nii('/Users/lucyowen/Desktop/try_nii_high_res')
# fill in the missing timeseries data with original set of locations
reconstructed_bo = model.predict(bo, nearest_neighbor = False)

reconstructed_bo_nn_40 = model.predict(bo, nearest_neighbor = True, match_threshold=40)


# print out info on new brain object
reconstructed_bo.info()

# save as nifti
reconstructed_nifti = reconstructed_bo.to_nii()



