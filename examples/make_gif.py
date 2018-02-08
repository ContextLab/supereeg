# -*- coding: utf-8 -*-
"""
=============================
Make gif
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.  We then convert the reconstruction to a nifti and convert 500 consecutive timepoints
to .png files and then compile as a gif.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# load
import supereeg as se


# load example data
bo = se.load('example_data')

# load example model
model = se.load('example_model')

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo = model.predict(bo)

# print out info on new brain object
reconstructed_bo.info()

# convert to nifti
reconstructed_nifti = reconstructed_bo.to_nii()

# make gif, default time window is 1000 to 1500, but you can specifiy
# se.make_gif_pngs(reconstructed_nifti, result_dir='/your/path/to/gif')
