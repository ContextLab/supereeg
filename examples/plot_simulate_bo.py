# -*- coding: utf-8 -*-
"""
=============================
Simulate brain object
=============================

In this example, we demonstrate the simulate brain object function.
First, we'll load in some example locations. Then we'll simulate 1
brain object specifying a noise parameter and the correlational structure
(a toeplitz matrix). We'll then subsample 10 locations from the original brain object.
"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se
import pandas as pd

# load example model to get locations
locs = se.load('example_locations')

# convert to pandas
locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

# simulate brain object
bo = se.simulate_bo(n_samples=1000, sample_rate=1000, cov='toeplitz', locs=locs, noise =.3)

# brain object locations, 10 sampled
sub_locs = locs.sample(10).sort_values(['x', 'y', 'z'])

# parse brain object to create synthetic patient data
data = bo.data.iloc[:, sub_locs.index]

# create synthetic patient
bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

# plot sample patient locations
bo_sample.plot_locs()

# plot sample patient data
bo_sample.plot_data()