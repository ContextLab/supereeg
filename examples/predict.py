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

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo = model.predict(bo)

# but pasing nearest_neighbor=False will fill in the missing timeseries data using the original set of locations
reconstructed_bo_no_nn = model.predict(bo, nearest_neighbor = False)

# you can also set your match threshold to a specific value
# this will match electrode to voxel if within 40 mms
reconstructed_bo_nn_40 = model.predict(bo, nearest_neighbor = True, match_threshold=40)


# another default is force_update=False.  if set to True, the model will update with the subject covariance matrix
reconstructed_bo_fu = model.predict(bo, nearest_neighbor = False, force_update=True)

# print out info on new brain object
reconstructed_bo.info()

# you can also see which locations were observed and which were predicted
reconstructed_bo.label()

# save as nifti
reconstructed_nifti = reconstructed_bo.to_nii()



