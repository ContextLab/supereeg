import pytest
import superEEG as se
import numpy as np
import pandas as pd
from superEEG._helpers.stats import *
from scipy.stats import kurtosis
import seaborn as sns

locs = se.load('example_locations')
# number of timeseries samples
n_samples = 1000
# number of subjects
n_subs = 5
# number of electrodes
n_elecs = 10
# full brain object to parse and compare
bo_full = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=locs)
# create brain object from subset of locations
sub_locs = bo_full.locs.iloc[160:]
sub_data = bo_full.data.iloc[:, sub_locs.index]
bo = se.Brain(data=sub_data.as_matrix(), locs=sub_locs, sample_rate=1000)

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

def test_expand_corrmat_fit():

    sub_corrmat = get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)
    sub_corrmat = r2z(sub_corrmat)
    weights = rbf(test_model.locs, bo.locs)
    expanded_num_f, expanded_denom_f = expand_corrmat_fit(sub_corrmat, weights)

    assert isinstance(expanded_num_f, np.ndarray)
    assert isinstance(expanded_denom_f, np.ndarray)
    assert np.shape(expanded_num_f)[0] == test_model.locs.shape[0]

def test_expand_corrmat_predict():

    sub_corrmat = get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)
    sub_corrmat = r2z(sub_corrmat)
    weights = rbf(test_model.locs, bo.locs)
    expanded_num_p, expanded_denom_p = expand_corrmat_predict(sub_corrmat, weights)

    assert isinstance(expanded_num_p, np.ndarray)
    assert isinstance(expanded_denom_p, np.ndarray)
    assert np.shape(expanded_num_p)[0] == test_model.locs.shape[0]

def test_expand_corrmats_same():

    sub_corrmat = get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0) # <- possible failpoint
    sub_corrmat_z = r2z(sub_corrmat)
    weights = rbf(test_model.locs, bo.locs)

    expanded_num_p, expanded_denom_p = expand_corrmat_predict(sub_corrmat_z, weights)
    model_corrmat_p = np.divide(expanded_num_p, expanded_denom_p)
    expanded_num_f, expanded_denom_f = expand_corrmat_predict(sub_corrmat_z, weights)
    model_corrmat_f = np.divide(expanded_num_f, expanded_denom_f)


    np.fill_diagonal(model_corrmat_f, 0)
    np.fill_diagonal(model_corrmat_p, 0)

    s = test_model.locs.shape[0]-bo.locs.shape[0]
    print(s)
    Kba_p = model_corrmat_p[:s,s:]
    Kba_f = model_corrmat_f[:s, s:]
    Kaa_p = model_corrmat_p[s:,s:]
    Kaa_f = model_corrmat_f[s:, s:]

    assert isinstance(Kaa_p, np.ndarray)
    assert isinstance(Kaa_f, np.ndarray)
    assert np.allclose(Kaa_p, Kaa_f)
    assert np.allclose(Kba_p, Kba_f)

def test_reconstruct():

    recon_test = test_model.predict(bo)
    actual_test = bo_full.data.iloc[:, recon_test.locs.index]

    mo = test_model.update(bo)
    model_corrmat_x = np.divide(mo.numerator, mo.denominator)
    model_corrmat_x = z2r(model_corrmat_x)
    np.fill_diagonal(model_corrmat_x, 0)
    recon_data = reconstruct_activity(bo, model_corrmat_x)
    corr_vals = corr_column(actual_test.as_matrix(), recon_test.data.as_matrix())
    assert isinstance(recon_data, np.ndarray)
    assert np.allclose(recon_data, recon_test.data)
    assert corr_vals.mean() >.7

### better way to test accuracy??


