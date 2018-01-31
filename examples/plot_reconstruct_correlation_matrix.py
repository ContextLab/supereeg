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

locs=pd.DataFrame(locs, columns=['x', 'y', 'z'])
# simulate correlation matrix
R = se.create_cov(cov='toeplitz', n_elecs=len(locs))

# n_samples
n_samples = 1000

# initialize subplots
f, axarr = plt.subplots(4, 4)

# loop over simulated subjects size
for isub, n_subs in enumerate([10, 25, 50, 100]):

    # loop over simulated electrodes
    for ielec, n_elecs in enumerate([10, 25, 50, 100]):

        # simulate brain objects for the model
        model_bos = [se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=locs, sample_locs=n_elecs, cov='toeplitz') for x in
                     range(n_subs)]

        # create the model object
        model = se.Model(data=model_bos, locs=locs)

        # plot it
        sns.heatmap(np.divide(model.numerator, model.denominator), ax=axarr[isub, ielec], yticklabels=False,
                    xticklabels=False, cmap='RdBu_r', cbar=False, vmin=0, vmax=3)

        # set the title
        axarr[isub, ielec].set_title(str(n_subs) + ' Subjects, ' + str(n_elecs) + ' Electrodes')

plt.tight_layout()
plt.show()
