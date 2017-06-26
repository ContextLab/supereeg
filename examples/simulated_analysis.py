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
from numpy import inf
from scipy.stats import zscore

# n_samples
n_samples = 1000

# load example model to get locations
model = se.load('example_model')

# get the locations
locs = model.locs

### to we convert to z before creating the synthetic data?

### if yes:

# # create a fake model
# temp = r2z(scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]))
# temp[temp == inf] = r2z(.999999)
# # temp[temp == inf] = 0 # converting the diagnol to 0s gives an indefinete matrix - cant use cholesky
# model = se.Model(data=temp, locs=locs)
#
# #model = se.Model(data=scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]), locs=locs)
# model = se.Model(data=model.data, locs=locs)

### if no:
model = se.Model(data=scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]), locs=locs)
# create a random multivariate distribution
rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)

# multiply by the model
bo = se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(model.data)), locs=locs)

#temp = r2z(model.data)
#temp[temp == inf] = 0
#model.data = temp
# create a brain object that is a subsample of the full data
#bo_sub = se.Brain(data=bo.data.iloc[:, range(1,len(locs), 10)], locs=bo.locs.iloc[range(1,len(locs), 10), :])
bo_sub = se.Brain(data=bo.data.iloc[:, range(1,len(locs))], locs=bo.locs.iloc[range(1,len(locs)), :])


# model.data = z2r(model.data)
# reconstructed = model.predict(bo_sub)
# expected = zscore(bo.data)


def reconstruct_activity(bo, K):
    """
    """
    s = K.shape[0]-bo.locs.shape[0]
    Kba = K[:s,s:]
    Kaa = K[s:,s:]
    Y = zscore(bo.get_data())
    return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)


reconstructed_activity = reconstruct_activity(bo_sub, model.data.as_matrix())