# -*- coding: utf-8 -*-
"""
=============================
Model reconstruction by number of subjects and electrodes
=============================

In this example, we will parametrically vary how many subjects and how many
electrodes per subject are used to create the model.  First, we load in some
example locations.  Then, we simulate a correlation matrix (toeplitz) to impose
on the simulated subject data. Finally, we loop over number of subjects and
number of randomly chosen electrodes and plot the model at each iteration. As
the figure shows, the more subjects and electrodes, the better then recovery of
the true model.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import scipy
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import superEEG as se

# load example model to get locations
locs = se.load('example_locations')

# simulate correlation matrix
R = scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1])

# n_samples
n_samples = 1000

# initialize subplots
f, axarr = plt.subplots(4, 4)

# loop over simulated subjects size
for isub, n_subs in enumerate([10, 25, 50, 100]):

    # loop over simulated electrodes
    for ielec, n_elecs in enumerate([10, 25, 50, 100]):

        # initialize data list
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

        # plot it
        sns.heatmap(np.divide(model.numerator,model.denominator), ax=axarr[isub,ielec], yticklabels=False, xticklabels=False, cmap='RdBu_r', cbar=False, vmin=0, vmax=3)

        # set the title
        axarr[isub,ielec].set_title(str(n_subs) + ' Subjects, ' + str(n_elecs) + ' Electrodes')

sns.plt.tight_layout()
sns.plt.show()
