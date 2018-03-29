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

# sample 10 locations, and get indices
sub_locs = locs.sample(10, replace=False).sort_values(['x', 'y', 'z'])

R = se.create_cov(cov='random', n_elecs=len(sub_locs))
toe_model = se.Model(data=R, locs=sub_locs)

bo_s = toe_model.predict(bo, nearest_neighbor=False)

# sample 10 locations, and get indices
sub_locs = locs.sample(10).sort_values(['x', 'y', 'z']).index.values.tolist()

# index brain object to get sample patient
bo_sample = bo[: ,sub_locs]

# plot sample patient locations
bo_sample.plot_locs()

# plot sample patient data
bo_sample.plot_data()
