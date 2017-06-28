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

        print(str(n_subs) + ' Subjects, ' + str(n_elecs) + ' Electrodes')

sns.plt.show()
