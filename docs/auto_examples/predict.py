# -*- coding: utf-8 -*-
"""
=============================
Predict unknown location
=============================

In this example, we load in a single subject example, remove electrodes that
exceed a kurtosis threshold, load a model, and predict activity at all
model locations.  We then convert the reconstruction to a nifti.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se


# load example data
bo = se.load('example_data')

# load example model
model = se.load('example_model')

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo = model.predict(bo)

# print out info on new brain object
reconstructed_bo.info()

# save as nifti
reconstructed_nifti = reconstructed_bo.to_nii()
