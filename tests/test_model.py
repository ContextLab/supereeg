# -*- coding: utf-8 -*-

import pytest
import superEEG as se
import numpy as np
import scipy
import pandas as pd
from superEEG._helpers.stats import rbf
import seaborn as sns
from sklearn import datasets

# load example model to get locations
locs = se.load('example_locations')
# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 3
# number of electrodes
n_elecs = 10
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs)

# make tests for attributes

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

def test_update():
    model = se.Model(data=data[1:3], locs=locs)
    mo = model.update(data[0])
    print(test_model.n_subs)
    print(mo.n_subs)
    assert isinstance(mo, se.Model)
    assert np.allclose(mo.numerator, test_model.numerator)
    assert np.allclose(mo.denominator, test_model.denominator)


