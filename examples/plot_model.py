# -*- coding: utf-8 -*-
"""
=============================
Load and plot a model
=============================

Here we load the example model, and then plot it along with the locations.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se

# load example model
#model = se.load('example_model')

model = se.load('pyFR_k10r20_6mm')
# plot it
model.plot_data(xticklabels=False, yticklabels=False)

# plot locations
model.plot_locs()