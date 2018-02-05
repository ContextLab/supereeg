# -*- coding: utf-8 -*-
"""
=============================
Explore labels
=============================

In this example, we load in a single subject example, load a model, and predict activity at all
model locations. We then parse the values based on the labels.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import superEEG as se
import copy as copy
import pandas as pd

# load example data
bo = se.load('example_data')

# load example model
model = se.load('example_model')

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo = model.predict(bo)

# find the observed indices
obs_inds = [i for i, x in enumerate(reconstructed_bo.label) if x == 'observed']

# make a copy of the brain object
o_bo = copy.copy(reconstructed_bo )

# replace fields with indexed data and locations
o_bo.data = pd.DataFrame(o_bo.get_data()[obs_inds, :])
o_bo.locs = pd.DataFrame(o_bo.get_locs()[obs_inds], columns=['x', 'y', 'z'])

# plot the original locations
bo.plot_locs()

# plot the nearest voxel used in the reconstruction
o_bo.plot_locs()
