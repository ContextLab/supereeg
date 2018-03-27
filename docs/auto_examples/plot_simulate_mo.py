# -*- coding: utf-8 -*-
"""
=============================
Simulate model object
=============================

In this example, we simulate 3 brain objects using a subset of 10
locations. We will impose a correlational structure (a toeplitz matrix) on
our simulated brain objects.  Then, we will create a model from these brain
objects and plot it.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se

# simulate 100 locations
locs = se.simulate_locations(n_elecs=100)

# simulate correlation matrix
R = se.create_cov(cov='toeplitz', n_elecs=len(locs))

# create list of simulated brain objects
model_bos = [se.simulate_model_bos(n_samples=1000, sample_rate=1000, cov=R,
                                   locs=locs, sample_locs=10) for x in range(3)]

# create model from subsampled gray locations
model = se.Model(model_bos, locs=locs)

# plot the model
model.plot_data()
