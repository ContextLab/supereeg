import pytest
import superEEG as se
import numpy as np

def test_load_example_data():
    bo = se.load('example_data')
    assert isinstance(bo, se.Brain)

def test_load_example_model():
    model = se.load('example_model')
    assert isinstance(model, se.Model)

def test_load_example_locations():
    locs = se.load('example_locations')
    assert isinstance(locs, np.ndarray)

def test_load_nifti():
    bo = se.load('example_nifti')
    assert isinstance(bo, se.Brain)
