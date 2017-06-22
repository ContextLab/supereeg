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
from superEEG._helpers.stats import r2z
from numpy import inf

# n_samples
n_samples = 1000

# load example model to get locations
model = se.load('example_model')

# get the locations
locs = model.locs

# create a fake model
temp = r2z(scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]))
temp[temp == inf] = r2z(.999999)
model = se.Model(data=temp, locs=locs)

model = se.Model(data=scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]), locs=locs)
# create a random multivariate distribution
rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)

# multiply by the model
bo = se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(model.data)), locs=locs)

# create a brain object that is a subsample of the full data
#bo_sub = se.Brain(data=bo.data.iloc[:, range(1,len(locs), 10)], locs=bo.locs.iloc[range(1,len(locs), 10), :])
bo_sub = se.Brain(data=bo.data.iloc[:, range(0,len(locs)-1)], locs=bo.locs.iloc[range(0,len(locs)-1), :])

reconstructed = model.predict(bo_sub)