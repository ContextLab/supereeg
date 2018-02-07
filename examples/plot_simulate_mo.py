# -*- coding: utf-8 -*-
"""
=============================
Simulate model object
=============================

Then we simulate 3 brain objects using a subset of 10
locations. We will simulate correlational structure (a toeplitz matrix) to impose on
our simulated data. However, this could also also be accomplished by specififying
cov='toeplitz' to create the model.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import pandas as pd
import supereeg as se


# load example model to get locations
locs = se.load('example_locations')

# convert to pandas
locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

# simulate correlation matrix
R = se.create_cov(cov='toeplitz', n_elecs=len(locs))

# create list of simulated brain objects
model_bos = [se.simulate_model_bos(n_samples=1000, sample_rate=1000, cov=R, locs=locs, sample_locs=10) for x in range(3)]

# create model from subsampled gray locations
model = se.Model(model_bos, locs=locs)

# plot the model
model.plot()