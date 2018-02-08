from builtins import range
import pytest
import supereeg as se
import numpy as np
import pandas as pd
from supereeg.helpers import _corr_column


locs = se.load('example_locations')
# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 3
# number of electrodes
n_elecs = 10
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs)
R = se.create_cov('random', len(locs))



recon_1 = np.matrix([[ 0.82919399,  0.3047186 ,  0.82919399,  0.94488569, -0.1156917 ],
                     [-0.39653203, -0.14572065, -0.39653203,  0.77469247, -1.1712245 ],
                     [-0.16080707, -0.05909462, -0.16080707,  0.72329345, -0.88410052],
                     [-0.63032456, -0.23163653, -0.63032456, -1.2276946 ,  0.59737003],
                     [ 0.35846968,  0.1317332 ,  0.35846968, -1.21517701,  1.57364669]])


recon_2 = np.matrix([[-0.02616138, -0.65365414, -0.30856683, -0.49671408, -0.28675327],
                     [-0.07208702, -0.65861915, -0.36174338, -0.03937399, -0.79014146],
                     [-0.11282797, -2.24664523, -1.08602914, -1.47620904, -1.23670058],
                     [ 0.08643809,  1.16408297,  0.59381912,  0.48276444,  0.94744261],
                     [ 0.12463829,  2.39483555,  1.16252024,  1.52953267,  1.36615270]])


recon_3 = np.matrix([[  1.72363835e-01,   3.09339273e-10,  -1.95257063e-01, 1.62790162e-01,  -2.93614621e-01],
                     [ -2.51070375e+00,  -3.71561269e-09,  -5.72328657e-01,-1.95534562e+00,   1.57187889e+00],
                     [  1.06992272e+00,   1.54327013e-09,   4.17322838e-01, 8.12147750e-01,  -5.32536689e-01],
                     [  3.12442895e-01,   7.97543472e-10,  -1.37764262e+00, 4.19708205e-01,  -1.34274447e+00],
                     [  9.55974302e-01,   1.06545982e-09,   1.72790550e+00, 5.60699502e-01,   5.97016891e-01]])



def test_simulate_locations():
    locs = se.simulate_locations(10)
    assert isinstance(locs, pd.DataFrame)

def test_simulate_model_bos():
    bo = se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs=n_elecs)
    assert isinstance(bo, se.Brain)

def test_simulate_model_bos_distance():
    bo = se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs=n_elecs, cov='distance')
    assert isinstance(bo, se.Brain)

def test_simulate_model_bos_np_array_R():
    bo = se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs=n_elecs, cov=R)
    assert isinstance(bo, se.Brain)

def test_simulate_model_data_random():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, sample_locs=n_elecs, cov='random')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, cov='random')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_np_array_R():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, sample_locs=n_elecs, cov=R)
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_np_array_R_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, cov=R)
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_distance():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, sample_locs=n_elecs, cov='distance')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_model_data_distance_no_sample_locs():
    data, sub_locs = se.simulate_model_data(n_samples=10, locs=locs, cov='distance')
    assert isinstance(data, np.ndarray)
    assert isinstance(sub_locs, pd.DataFrame)

def test_simulate_bo():
    bo = se.simulate_bo(n_samples=10, sample_rate=1000, locs=locs)
    assert isinstance(bo, se.Brain)

def test_simulate_bo_no_locs():
    bo = se.simulate_bo(n_samples=10, sample_rate=1000)
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

    # set random seed to default and noise to 0
    random_seed = np.random.seed(123)
    noise = 0

    # load mini model
    gray = se.load('gray_mask_20mm_brain')

    # extract 20 locations
    gray_locs = gray.locs.iloc[:5]

    # create model from 10 locations
    mo_locs = gray_locs.sample(3, random_state=random_seed).sort_values(['x', 'y', 'z'])

    # create covariance matrix from random seed
    c = se.create_cov(cov='random', n_elecs=5)

    # pull out model from covariance matrix
    data = c[:, mo_locs.index][mo_locs.index, :]

    # create model from subsetted covariance matrix and locations
    model = se.Model(numerator=np.array(data), denominator=np.ones(np.shape(data)), locs=mo_locs,
                     n_subs=1)

    # create brain object from the remaining locations - first find remaining locations
    sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]

    # create a brain object with all gray locations
    bo = se.simulate_bo(n_samples=5, sample_rate=1000, locs=gray_locs, noise=noise, random_seed=random_seed)

    # parse brain object to create synthetic patient data
    data = bo.data.iloc[:, sub_locs.index]

    # put data and locations together in new sample brain object
    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    # actual = bo.data.iloc[:, unknown_ind]
    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.as_matrix(), recon.data.as_matrix())

    assert 1 >= corr_vals.mean() >= -1
    assert np.allclose(recon_1, recon.get_data())

def test_electrode_contingencies_2_subset():

    random_seed = np.random.seed(123)

    noise = 0

    gray = se.load('gray_mask_20mm_brain')

    # extract locations
    gray_locs = gray.locs.iloc[:5]

    mo_locs = gray_locs

    c = se.create_cov(cov='random', n_elecs=5)

    data = c[:, mo_locs.index][mo_locs.index, :]

    model = se.Model(numerator=np.array(data), denominator=np.ones(np.shape(data)), locs=mo_locs, n_subs=1)

    # create brain object from the remaining locations - first find remaining locations
    sub_locs = mo_locs.sample(2, random_state=random_seed).sort_values(['x', 'y', 'z'])

    # create a brain object with all gray locations
    bo = se.simulate_bo(n_samples=5, sample_rate=1000, locs=gray_locs, noise=noise, random_seed=random_seed)

    # parse brain object to create synthetic patient data
    data = bo.data.iloc[:, sub_locs.index]

    # put data and locations together in new sample brain object
    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.as_matrix(), recon.data.as_matrix())

    assert np.allclose(recon_2, recon.get_data())
    assert 1 >= corr_vals.mean() >= -1

def test_electrode_contingencies_3_locations_can_subset():
    random_seed = np.random.seed(123)
    noise = 0

    # load mini model
    gray = se.load('gray_mask_20mm_brain')

    # extract 20 locations
    gray_locs = gray.locs.iloc[:5]

    # create model from 10 locations
    mo_locs = gray_locs.sample(4, random_state=random_seed).sort_values(['x', 'y', 'z'])

    # create covariance matrix from random seed
    c = se.create_cov(cov='random', n_elecs=5)

    # pull out model from covariance matrix
    data = c[:, mo_locs.index][mo_locs.index, :]

    # create model from subsetted covariance matrix and locations
    model = se.Model(numerator=np.array(data), denominator=np.ones(np.shape(data)), locs=mo_locs,
                     n_subs=1)

    # create brain object from the remaining locations - first find remaining locations
    sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]

    sub_locs = sub_locs.append(gray_locs.sample(1, random_state=random_seed).sort_values(['x', 'y', 'z']))

    # create a brain object with all gray locations
    bo = se.simulate_bo(n_samples=5, sample_rate=1000, locs=gray_locs, noise=noise, random_seed=random_seed)

    # parse brain object to create synthetic patient data
    data = bo.data.iloc[:, sub_locs.index]

    # put data and locations together in new sample brain object
    bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    # actual = bo.data.iloc[:, unknown_ind]
    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.as_matrix(), recon.data.as_matrix())

    assert 1 >= corr_vals.mean() >= -1
    assert np.allclose(recon_3, recon.get_data())
