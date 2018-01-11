import pytest
import superEEG as se
import numpy as np
import pandas as pd
from superEEG._helpers.stats import *
from scipy.stats import kurtosis

locs = se.load('example_locations')
# number of timeseries samples
n_samples = 1000
# number of subjects
n_subs = 5
# number of electrodes
n_elecs = 10
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs)

def test_apply_by_file_index():

    def aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    kurts_1 = apply_by_file_index(data[0], kurtosis, aggregate)
    assert isinstance(kurts_1, np.ndarray)

def test_kurt_vals():
    kurts_2 = kurt_vals(data[0])
    assert isinstance(kurts_2, np.ndarray)


def test_kurt_vals_compare():
    def aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    kurts_1 = apply_by_file_index(data[0], kurtosis, aggregate)
    kurts_2 = kurt_vals(data[0])
    assert np.allclose(kurts_1,kurts_2)


def test_get_corrmat():
    corrmat = get_corrmat(data[0])
    assert isinstance(corrmat, np.ndarray)

def test_int_z2r():
    z = 1
    test_val = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    input_val = z2r(z)
    assert isinstance(input_val, (float, int))
    assert test_val==input_val

def test_array_z2r():
    z = [1,2,3]
    test_val = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    input_val = z2r(z)
    assert isinstance(input_val, np.ndarray)
    assert np.allclose(test_val,input_val)

def r2z_z2r():
    z = np.array([1,2,3])
    input_val = r2z(z2r(z))
    assert isinstance(input_val, (int, np.ndarray))
    assert z==input_val


def test_int_r2z():
    r = .1
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    input_val = r2z(r)
    assert isinstance(input_val, (float, int))
    assert test_val==input_val

def test_array_r2z():
    r = np.array([.1,.2,.3])
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    input_val = r2z(r)
    assert isinstance(input_val, np.ndarray)
    assert np.allclose(test_val,input_val)

def test_rbf():
    weights = rbf(locs, locs[:10])
    weights_same = rbf(locs[:10], locs[:10], 1)
    print (weights_same)
    print(np.zeros(np.shape(weights_same)))
    assert isinstance(weights, np.ndarray)
    assert np.allclose(weights_same, np.eye(np.shape(weights_same)[0]))

def test_tal2mni():
    tal_vals = tal2mni(locs)
    assert isinstance(tal_vals, np.ndarray)




