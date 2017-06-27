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

# create a fake model
full_model = se.Model(data=scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1]), locs=locs)

# create a random multivariate distribution
rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)

# multiply by the model to create the synthetic full brain activity
bo = se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(full_model.data)), locs=locs)

# indices: subset of 10 from full location for the synthetic subject data and the rest for the synthetic model
locs_inds = range(0,len(locs))
sub_inds = locs_inds[0::5]
model_inds = list(set(locs_inds)-set(sub_inds))


# create a brain object that is a subsample of the full data - this is the synthetic subject data
bo_sub = se.Brain(data=bo.data.iloc[:, sub_inds], locs=bo.locs.iloc[sub_inds, :])

# create a new model that is a subsample of the full model - this is the synthetic model
model = se.Model(data=full_model.data.as_matrix()[:, model_inds][model_inds], locs=full_model.locs.iloc[model_inds, :])

temp = r2z(model.data)
temp[temp == inf] = 0
model.data = temp
reconstructed = model.predict(bo_sub)
expected = zscore(bo.data.iloc[:, model_inds])
import seaborn as sb
sb.jointplot(reconstructed.data.iloc[:,0], expected[:, 0])

### below works:

bo_sub = se.Brain(data=bo.data.iloc[:, range(1,len(locs))], locs=bo.locs.iloc[range(1,len(locs)), :])

def reconstruct_activity(bo, K):
    """
    """
    s = K.shape[0]-bo.locs.shape[0]
    Kba = K[:s,s:]
    Kaa = K[s:,s:]
    Y = zscore(bo.get_data())
    return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)


reconstructed_activity = reconstruct_activity(bo_sub, model.data.as_matrix())

import seaborn as sb
sb.jointplot(reconstructed_activity, bo.data[0])