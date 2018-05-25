# -*- coding: utf-8 -*-

from __future__ import print_function
#from builtins import range
import supereeg as se
import numpy as np
import scipy
import pytest

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
n_subs = 6
# number of electrodes
n_elecs = 5
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data[0:3], locs=locs, rbf_width=20)

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
    print(data[0].n_secs)
    assert isinstance(bo, se.Brain)

def test_model_predict_nn():
    print(data[0].n_secs)
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
    mo = se.Model([model, data[0]])
    assert isinstance(mo, se.Model)
    assert np.allclose(mo.numerator.real, test_model.numerator.real)
    assert np.allclose(mo.numerator.imag, test_model.numerator.imag)
    assert np.allclose(mo.denominator, test_model.denominator)

def test_create_model_str():
    model = se.Model('example_data')
    assert isinstance(model, se.Model)

def test_create_model_model():
    mo = se.Model(data=data[1:3], locs=locs)
    model = se.Model(mo)
    assert isinstance(model, se.Model)

def test_model_update_inplace():
    mo = se.Model(data=data[1:3], locs=locs)
    mo = mo.update(data[0])
    assert mo is None

def test_model_update_not_inplace():
    mo = se.Model(data=data[1:3], locs=locs)
    mo = mo.update(data[0], inplace=False)
    assert isinstance(mo, se.Model)

def test_model_update_with_model():
    mo = se.Model(data=data[1:3], locs=locs)
    mo = mo.update(mo, inplace=False)
    assert isinstance(mo, se.Model)

def test_model_update_with_model_and_bo():
    mo = se.Model(data=data[1:3], locs=locs)
    mo = se.Model([mo, data[0]])
    assert isinstance(mo, se.Model)

def test_model_update_with_array():
    mo = se.Model(data=data[1:3], locs=locs)
    d = np.random.rand(*mo.numerator.shape)
    mo = se.Model([mo, d], locs=mo.get_locs())
    assert isinstance(mo, se.Model)

#This syntax is ambiguous and no longer supported
#def test_model_update_with_smaller_array():
#    mo = se.Model(data=data[1:3], locs=locs)
#    d = np.random.rand(3,3)
#    with pytest.raises(ValueError):
#        mo = se.Model([mo, d])

def test_model_get_model():
    mo = se.Model(data=data[1:3], locs=locs)
    m = mo.get_model()
    assert isinstance(m, np.ndarray)

def test_model_get_slice():
    mo = se.Model(data=data[1:3], locs=locs)
    inds = [0, 1]
    s = mo.get_slice(inds)
    assert(type(s) == se.Model)
    s_model = s.get_model()
    assert s_model.shape[0] == s_model.shape[1]
    assert s_model.shape[0] == len(inds)
    assert s_model.shape[0] == len(inds)

    mo.get_slice(inds, inplace=True)
    assert(type(mo) == se.Model)
    mo_model = mo.get_model()
    assert mo_model.shape[0] == mo_model.shape[1]
    assert mo_model.shape[0] == len(inds)
    assert mo_model.shape[0] == len(inds)

def test_model_add():
    mo1 = se.Model(data=data[0:3], locs=locs)
    mo2 = se.Model(data=data[3:6], locs=locs)
    mo3 = mo1 + mo2

    mo1_model = mo1.get_model()
    mo2_model = mo2.get_model()
    mo3_model = mo3.get_model()
    assert np.allclose(mo1_model.shape, mo2_model.shape)
    assert np.allclose(mo2_model.shape, mo3_model.shape)
    assert mo1_model.shape[0] == mo1_model.shape[1]

    assert mo3.n_subs == mo1.n_subs + mo2.n_subs

    mo3_alt = se.Model(data=data[0:6], locs=locs)
    assert np.allclose(mo3.numerator.real, mo3_alt.numerator.real)
    assert np.allclose(mo3.numerator.imag, mo3_alt.numerator.imag)
    assert np.allclose(mo3.denominator, mo3_alt.denominator)

#subtraction is not working; removing functionality until fixed
# def test_model_subtract():
#     mo1 = se.Model(data=data[0:3], locs=locs)
#     mo2 = se.Model(data=data[3:6], locs=locs)
#     mo3 = mo1 - mo2
#
#     mo1_model = mo1.get_model()
#     mo2_model = mo2.get_model()
#     mo3_model = mo3.get_model()
#     assert np.allclose(mo1_model.shape, mo2_model.shape)
#     assert np.allclose(mo2_model.shape, mo3_model.shape)
#     assert mo1_model.shape[0] == mo1_model.shape[1]
#
#     assert mo3.n_subs == mo1.n_subs - mo2.n_subs
#
#     mo2_recon = mo3 - mo1
#     assert np.allclose(mo2.numerator.real, mo2_recon.numerator.real)
#     assert np.allclose(mo2.numerator.imag, mo2_recon.numerator.imag)
#     assert np.allclose(mo2.denominator, mo2_recon.denominator)
#
#     mo1_recon = mo3 - mo2
#     assert np.allclose(mo1.numerator.real, mo1_recon.numerator.real)
#     assert np.allclose(mo1.numerator.imag, mo1_recon.numerator.imag)
#     assert np.allclose(mo1.denominator, mo1_recon.denominator)
