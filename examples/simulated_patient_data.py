# -*- coding: utf-8 -*-
"""
=============================
Simulate data
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se
import scipy
import numpy as np
from superEEG._helpers.stats import r2z, z2r
from superEEG._helpers.bookkeeping import slice_list
from numpy import inf
from scipy.stats import zscore
import os
import pandas as pd

# n_samples
n_samples = 1000

# load example model to get locations
with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
    locs = np.load(handle)

# get the locations

R = scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1])

count = 0
patient_inds = slice_list(range(0, len(locs)), 13)

for p in patient_inds:

    count += 1
    rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)
    bo = se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(R))[:,p], locs=pd.DataFrame(locs[p,:], columns=['x', 'y', 'z']))

    bo.to_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/synthetic_' + str(count))
