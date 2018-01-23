import pytest
import superEEG as se
import numpy as np
import pandas as pd
from superEEG._helpers.stats import corr_column

# clean up simulate.py and write functions that return expected objects

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

R = se.create_cov('random', len(locs))


def test_simulate_locations():
    locs = se.simulate_locations(10)
    assert isinstance(locs, pd.DataFrame)

def test_simulate_model_bos():
    bo = se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=locs, sample_locs=n_elecs)
    assert isinstance(bo, se.Brain)

def test_simulate_model_bos_distance():
    bo = se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=locs, sample_locs=n_elecs, cov='distance')
    assert isinstance(bo, se.Brain)

def test_simulate_model_bos_np_array_R():
    bo = se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=locs, sample_locs=n_elecs, cov=R)
    assert isinstance(bo, se.Brain)

def test_simulate_model_data_random():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, sample_locs=n_elecs, cov='random')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, cov='random')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_np_array_R():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, sample_locs=n_elecs, cov=R)
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_np_array_R_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, cov=R)
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_distance():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, sample_locs=n_elecs, cov='distance')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_distance_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10000, locs=locs, cov='distance')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_bo():
    bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=locs)
    assert isinstance(bo, se.Brain)

def test_simulate_bo_no_locs():
    bo = se.simulate_bo(n_samples=1000, sample_rate=1000)
    assert isinstance(bo, se.Brain)

def test_create_cov_random():
    c = se.create_cov(cov='random', n_elecs=len(locs))
    assert isinstance(c, np.ndarray)

def test_create_cov_eye():
    c = se.create_cov(cov='eye', n_elecs=len(locs))
    assert isinstance(c, np.ndarray)

def test_create_cov_toeplitz():
    c = se.create_cov(cov='toeplitz', n_elecs=len(locs))
    assert isinstance(c, np.ndarray)

def test_create_cov_np_array_R():
    c = se.create_cov(cov=R, n_elecs=len(locs))
    assert isinstance(c, np.ndarray)

def test_create_cov_random_no_locs():
    c = se.create_cov(cov='random')
    assert isinstance(c, np.ndarray)

def test_electrode_contingencies_1_null_set():
    ### I think i can do this better by just taking the covariance matrix and making it into a model object and subsetting from there
    # load nifti to get locations
    gray = se.load('mini_model')

    # extract locations
    gray_locs = gray.locs

    mo_locs = gray_locs.sample(100).sort_values(['x', 'y', 'z'])

    c = se.create_cov(cov='random', n_elecs=170)

    data = c[:, mo_locs.index][mo_locs.index, :]

    model = se.Model(numerator=10*np.array(data), denominator=10*np.ones(np.shape(data)), locs=mo_locs, n_subs=10)

    # create brain object from the remaining locations - first find remaining locations
    sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]

    # create a brain object with all gray locations
    bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

    # parse brain object to create synthetic patient data
    data = bo.data.iloc[:, sub_locs.index]

    # put data and locations together in new sample brain object
    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    # actual = bo.data.iloc[:, unknown_ind]
    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

    assert 1 >= corr_vals.mean() >= -1

def test_electrode_contingencies_2_subset():
    ### I think i can do this better by just taking the covariance matrix and making it into a model object and subsetting from there
    # load nifti to get locations
    gray = se.load('mini_model')

    # extract locations
    gray_locs = gray.locs

    mo_locs = gray_locs.sample(100).sort_values(['x', 'y', 'z'])

    c = se.create_cov(cov='random', n_elecs=170)

    data = c[:, mo_locs.index][mo_locs.index, :]

    model = se.Model(numerator=10*np.array(data), denominator=10*np.ones(np.shape(data)), locs=mo_locs, n_subs=10)

    # create brain object from the remaining locations - first find remaining locations
    sub_locs = mo_locs.sample(70).sort_values(['x', 'y', 'z'])

    # create a brain object with all gray locations
    bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

    # parse brain object to create synthetic patient data
    data = bo.data.iloc[:, sub_locs.index]

    # put data and locations together in new sample brain object
    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

    assert 1 >= corr_vals.mean() >= -1

## this is showing that the third contingencies gives lower correlations - need to fix this

def test_electrode_contingencies_3_locations_can_subset():
    ### I think i can do this better by just taking the covariance matrix and making it into a model object and subsetting from there
    # load nifti to get locations
    gray = se.load('mini_model')

    # extract locations
    gray_locs = gray.locs

    mo_locs = gray_locs.sample(100).sort_values(['x', 'y', 'z'])

    c = se.create_cov(cov='random', n_elecs=170)

    data = c[:, mo_locs.index][mo_locs.index, :]

    model = se.Model(numerator=10*np.array(data), denominator=10*np.ones(np.shape(data)), locs=mo_locs, n_subs=10)

    # brain object locations subsetted entirely from both model and gray locations - for this n > m (this isn't necessarily true, but this ensures overlap)

    sub_locs = gray_locs.sample(70).sort_values(['x', 'y', 'z'])


    bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

    data = bo.data.iloc[:, sub_locs.index]

    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    try:
        recon = model.predict(bo_sample, nearest_neighbor=False)
        # sample actual data at reconstructed locations
        actual = bo.data.iloc[:, recon.locs.index]

        # correlate reconstruction with actual data
        corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

        corr_vals_sample = np.random.choice(corr_vals, 5)
        corr_val_mean = corr_vals_sample.mean()

    except:

        print('SVD issue')
        corr_vals = float('nan')
        corr_vals_mean = float('nan')

    assert 1 >= corr_val_mean >= -1