import pytest
import superEEG as se
import numpy as np

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

# make tests for attributes

def test_simulate_data():
    model = se.Model(data=data[0], locs=locs)
    assert isinstance(model, se.Model)