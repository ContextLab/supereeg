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


##### _helpers/stats ########

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
    test_fun = z2r(z)
    assert isinstance(test_fun, np.ndarray)
    assert np.allclose(test_val,test_fun)

def r2z_z2r():
    z = np.array([1,2,3])
    test_fun = r2z(z2r(z))
    assert isinstance(test_fun, (int, np.ndarray))
    assert z==test_fun


def test_int_r2z():
    r = .1
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    test_fun = r2z(r)
    assert isinstance(test_fun, (float, int))
    assert test_val==test_fun

def test_array_r2z():
    r = np.array([.1,.2,.3])
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    test_fun = r2z(r)
    assert isinstance(test_fun, np.ndarray)
    assert np.allclose(test_val,test_fun)

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

def test_uniquerows():
    full_locs = np.concatenate((locs, locs[:10]), axis=0)
    test_fun = uniquerows(full_locs)
    assert isinstance(test_fun, np.ndarray)
    assert np.shape(test_fun)==np.shape(locs)

### still would like to have a better test than this:
def test_expand_corrmat_fit():
    sub_locs = locs[:10]
    mod_locs = locs[10:]
    R = se.create_cov('random', len(sub_locs))
    weights = rbf(mod_locs, sub_locs)
    expanded_num, expanded_denom = expand_corrmat_fit(R, weights)
    assert isinstance(expanded_num, np.ndarray)
    assert isinstance(expanded_denom, np.ndarray)
    assert np.shape(expanded_num)[0] == np.shape(mod_locs)[0]

def test_expand_corrmat_predict():
    sub_locs = locs[:10]
    mod_locs = locs[10:]
    R = se.create_cov('random', len(sub_locs))
    weights = rbf(mod_locs, sub_locs)
    expanded_num, expanded_denom = expand_corrmat_predict(R, weights)
    assert isinstance(expanded_num, np.ndarray)
    assert isinstance(expanded_denom, np.ndarray)
    assert np.shape(expanded_num)[0] == np.shape(mod_locs)[0]

def test_expand_corrmats_same():
    sub_locs = locs[:10]
    print(np.shape(sub_locs))
    mod_locs = locs[10:]
    print(np.shape(mod_locs))
    R = se.create_cov('random', len(sub_locs))
    weights = rbf(mod_locs, sub_locs)
    expanded_num_p, expanded_denom_p = expand_corrmat_predict(R, weights)
    model_corrmat_p = np.divide(expanded_num_p, expanded_denom_p)
    expanded_num_f, expanded_denom_f = expand_corrmat_predict(R, weights)
    model_corrmat_f = np.divide(expanded_num_f, expanded_denom_f)
    s = R.shape[0]-np.shape(sub_locs)[0]
    Kba_p = model_corrmat_p[:s,s:]
    Kba_f = model_corrmat_f[:s, s:]
    Kaa_p = model_corrmat_p[s:,s:]
    Kaa_f = model_corrmat_f[s:, s:]
    print(np.shape(model_corrmat_p))
    print(np.shape(Kaa_p))
    print(np.shape(Kaa_f))
    assert isinstance(Kaa_p, np.ndarray)
    assert np.allclose(Kaa_p, Kaa_f)

