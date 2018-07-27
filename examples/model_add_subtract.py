# -*- coding: utf-8 -*-
"""
=============================
Explore model add and subtract
=============================

In this example, we show you how to add and subtract models.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT
import supereeg as se
import numpy as np

# some example locations
locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21.,  63.,  -3.],
                 [ -1., -37.,  37.],
                 [ -1.,  23.,  17.],
                 [ 19., -57., -23.],
                 [ 19.,  23.,  -3.],
                 [ 39., -57.,  17.],
                 [ 39.,   3.,  37.],
                 [ 59., -17.,  17.]])


# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 6
# number of electrodes
n_elecs = 5
# simulate some brain objects
data = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = n_elecs, set_random_seed=123, noise=0) for x in range(n_subs)]
# create a model from the first 5 brain objects and another from 1 brain object
mo1 = se.Model(data=data[0:5], locs=locs, n_subs=5)
mo2 = se.Model(data=data[5:6], locs=locs, n_subs=1)

# adding the models
mo3 = mo1 + mo2

# plot the added model
mo3.plot_data()
# adding these models is the same as making a model from all 6 brain objects at once
mo3_alt = se.Model(data=data[0:6], locs=locs, n_subs=6)
# plot the alternate model
mo3_alt.plot_data()
# show that they're the same
assert np.allclose(mo3.get_model(), mo3_alt.get_model())
# show that the number of subjects is also added
assert(mo3.n_subs == mo1.n_subs + mo2.n_subs)

# you can also subtract models
mo2_sub = mo3 - mo1

# plot the subtracted model
mo2_sub.plot_data()
# plot the original
mo2.plot_data()
# show that subratracting mo1 from mo3 will equal mo2
assert np.allclose(mo2.get_model(), mo2_sub.get_model(), equal_nan=True)
# show that the number of subjects is also subtracted
assert(mo2_sub.n_subs == mo2.n_subs)
# subtraction also updates the meta field, changing stable from True to False
mo2.info()
mo2_sub.info()
# now that the new model is not stable, so you can't add anything to it
try:
    assert mo2_sub + mo3
except AssertionError:
    assert True == True

