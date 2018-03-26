# -*- coding: utf-8 -*-
"""
=============================
Explore labels
=============================

In this example, we load in a single subject example, load a model, and predict activity at all
model locations. We then slice the brain object based on the labels.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se
import copy as copy
import pandas as pd

# load example data
bo = se.load('example_data')

# plot original locations
bo.plot_locs()

# load example model
model = se.load('example_model')

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo_nn = model.predict(bo)
reconstructed_bo_nn.plot_locs()

reconstructed_bo = model.predict(bo, nearest_neighbor=False, force_update=True)

# plot the all reconstructed locations
reconstructed_bo.plot_locs()

# find the observed indices
obs_inds = [i for i, x in enumerate(reconstructed_bo.label) if x == 'observed']
#
# # make a copy of the brain object
# o_bo = copy.copy(reconstructed_bo)

# slice data at observed indices inplace
reconstructed_bo.get_slice(sample_inds=obs_inds, loc_inds=obs_inds, inplace=True)

# plot the nearest voxel used in the reconstruction
reconstructed_bo.plot_locs()
