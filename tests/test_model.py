# -*- coding: utf-8 -*-

from __future__ import print_function
from builtins import range
import pytest
import supereeg as se
import numpy as np
import scipy

# some example locations

locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21.,  63.,  -3.],
                 [ -1., -37.,  37.],
                 [ -1.,  23.,  17.],
                 [ 19., -57., -23.],
                 [ 19.,  23.,  -3.],
                 [ 39., -57.,  17.],
                 [ 39.,   3.,  37.],
                 [ 59., -17.,  17.]])


# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 3
# number of electrodes
n_elecs = 5
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs)


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
    bo = model.predict(data[0], nearest_neighbor=False)
    assert isinstance(bo, se.Brain)

def test_model_predict_nn():
    model = se.Model(data=data[0:2], locs=locs)
    bo = model.predict(data[0], nearest_neighbor=True)
    assert isinstance(bo, se.Brain)

def test_model_predict_nn_thresh():
    model = se.Model(data=data[0:2], locs=locs)
    bo = model.predict(data[0], nearest_neighbor=True, match_threshold=30)
    assert isinstance(bo, se.Brain)

def test_model_predict_nn_0():
    model = se.Model(data=data[0:2], locs=locs)
    bo_1 = model.predict(data[0], nearest_neighbor=True, match_threshold=0)
    bo_2 = model.predict(data[0], nearest_neighbor=False)
    assert isinstance(bo_1, se.Brain)
    assert np.allclose(bo_1.get_data(), bo_2.get_data())

def test_update():
    model = se.Model(data=data[1:3], locs=locs)
    mo = model.update(data[0])
    print(test_model.n_subs)
    print(mo.n_subs)
    assert isinstance(mo, se.Model)
    assert np.allclose(mo.numerator, test_model.numerator)
    assert np.allclose(mo.denominator, test_model.denominator)
