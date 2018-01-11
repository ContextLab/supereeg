import pytest
import superEEG as se
import numpy as np
import pandas as pd
from superEEG._helpers.stats import *

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
    apply_by_file_index()