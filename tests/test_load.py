import pytest
import superEEG as se
import numpy as np
import os

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)
locs = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10)
test_bo = se.Brain(data=data, locs=locs)
# with open(filepath + '.bo', 'wb') as f:
#     pickle.dump(self, f)
#     print('Brain object saved as pickle.')

# update these tests for the new loaded data

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

## this should be replaced with test_load_pyFR_k10r20_6mm()
# def test_load_pyFR_k10r20_8mm():
#     bo = se.load('pyFR_k10r20_8mm')
#     assert isinstance(bo, se.Model)

def test_load_pyFR_union():
    data = se.load('pyFR_union')
    assert isinstance(data, np.ndarray)

def test_load_mini_model():
    bo = se.load('mini_model')
    assert isinstance(bo, se.Brain)

def test_load_gray_mask_6mm_brain():
    bo = se.load('gray_mask_6mm_brain')
    assert isinstance(bo, se.Brain)


# def test_bo_load(tmpdir):
#     p = tmpdir.mkdir("sub").join("bo")
#     p.write(test_bo)
#     assert p.read() == "bo"
#     assert len(tmpdir.listdir()) == 1