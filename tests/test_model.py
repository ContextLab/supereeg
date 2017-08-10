# -*- coding: utf-8 -*-

import pytest
import superEEG as se
import numpy as np
import scipy
import pandas as pd
from superEEG._helpers.stats import get_expanded_corrmat, rbf
import seaborn as sns

# load example model to get locations
locs = se.load('example_locations')

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


def test_create_model_1bo():
    model = se.Model(data=data[0], locs=locs)
    assert isinstance(model, se.Model)

def test_create_model_2bo():
    model = se.Model(data=data[0:2], locs=locs)
    assert isinstance(model, se.Model)

def test_create_model_superuser():
    locs = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10)
    numerator = scipy.linalg.toeplitz(np.linspace(0,10,len(locs))[::-1])
    denominator = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=10)
    model = se.Model(numerator=numerator, denominator=denominator, locs=locs, n_subs=2)
    assert isinstance(model, se.Model)

def test_model_predict():
    model = se.Model(data=data[0:2], locs=locs)
    bo = model.predict(data[0])
    assert isinstance(bo, se.Brain)

### need to finish this test I think
def test_expand_corrmat():
    R = scipy.linalg.toeplitz(np.linspace(0, 1, 3)[::-1])
    model_locs = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    subject_locs = np.array([[0,0,3], [0,0,4]])
    weights = rbf(np.vstack([model_locs, subject_locs]), model_locs, width=2)
    fit_num, fit_denom = get_expanded_corrmat(R, weights)
