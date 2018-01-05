import pytest
import superEEG as se
import numpy as np
import pandas as pd

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

