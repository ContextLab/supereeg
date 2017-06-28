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

new_data = []

# loop over simulated subjects
for i in range(n_subs):

    # for each subject, randomly choose n_elecs electrode locations
    p = np.random.choice(range(len(locs)), n_elecs, replace=False)

    # generate some random data
    rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)

    # new brain object
    new_data.append(se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(R))[:,p], locs=pd.DataFrame(locs[p,:], columns=['x', 'y', 'z'])))

# update the model
new_model = model.update(new_data)

# initialize subplots
f, (ax1, ax2) = plt.subplots(1, 2)

# plot it and set the title
model.plot(ax=ax1, yticklabels=False, xticklabels=False, cmap='RdBu_r', cbar=True, vmin=0, vmax=1)
ax1.set_title('Before updating model: 10 subjects total')

# plot it and set the title
new_model.plot(ax=ax2, yticklabels=False, xticklabels=False, cmap='RdBu_r', cbar=True, vmin=0, vmax=1)
ax2.set_title('After updating model: 20 subjects total')

sns.plt.show()
