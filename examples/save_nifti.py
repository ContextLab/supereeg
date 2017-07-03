# -*- coding: utf-8 -*-
"""
=============================
Predict unknown location
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se
import seaborn as sns
import numpy as np
from nilearn import plotting

# load example data
bo = se.load('example_data')

bo.data = bo.data.loc[:1000,:]
bo.sessions = bo.sessions.loc[:1000]

import superEEG as se
import scipy
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load example model to get locations
with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
    locs = np.load(handle)

# simulate correlation matrix
R = scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1])

# number of timeseries samples
n_samples = 1000

# number of subjects
n_subs = 10

# number of electrodes
n_elecs = 20

data = []

# loop over simulated subjects
for i in range(n_subs):

    # for each subject, randomly choose n_elecs electrode locations
    p = np.random.choice(range(len(locs)), n_elecs, replace=False)

    # generate some random data
    rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)

    # impose R correlational structure on the random data, create the brain object and append to data
    data.append(se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(R))[:,p], locs=pd.DataFrame(locs[p,:], columns=['x', 'y', 'z'])))

# create the model object
model = se.Model(data=data, locs=locs)

# # load example model
# model = se.load('example_model')
#
# # fill in the missing timeseries data
bo = model.predict(bo)

# save as nifti
data = bo.to_nifti()

# Import image processing tool
from nilearn import image

# Compute the voxel_wise mean of functional images across time.
# Basically reducing the functional image from 4D to 3D
mean_haxby_img = image.index_img(data, 1)

# Visualizing mean image (3D)
# plotting.plot_glass_brain(mean_haxby_img, title='plot_glass_brain')
# plotting.show()

# sns.heatmap(np.divide(bo.locs, new_locs))
# sns.plt.show()
