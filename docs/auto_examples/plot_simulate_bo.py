# -*- coding: utf-8 -*-
"""
=============================
Simulating a brain object
=============================

In this example, we demonstrate the simulate_bo function.
First, we'll load in some example locations. Then we'll simulate 1
brain object specifying a noise parameter and the correlational structure
of the data (a toeplitz matrix). We'll then subsample 10 locations from the
original brain object.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se


# simulate 100 locations
locs = se.simulate_locations(n_elecs=100)

# simulate brain object
bo = se.simulate_bo(n_samples=1000, sample_rate=1000, cov='toeplitz', locs=locs, noise =.3)

# brain object locations, 10 sampled
sub_locs = locs.sample(10).sort_values(['x', 'y', 'z'])

# parse brain object to create synthetic patient data
bo_sample = bo.get_slice(loc_inds=sub_locs.index.values.tolist())

# plot sample patient locations
bo_sample.plot_locs()

# plot sample patient data
bo_sample.plot_data()
