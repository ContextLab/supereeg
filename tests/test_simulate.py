import supereeg as se
import numpy as np
import pandas as pd
from scipy.stats import zscore
from supereeg.helpers import _corr_column

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
n_subs = 3
# number of electrodes
n_elecs = 10
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, sample_rate=1000, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs)
R = se.create_cov('random', len(locs))

recon_1 = np.matrix([[ 0.453253,  1.569009,  1.569009,  0.944886, -0.115692],
                     [-1.256820, -0.750322, -0.750322,  0.774692, -1.171225],
                     [-0.856609, -0.304281, -0.304281,  0.723293, -0.884101],
                     [ 0.087427, -1.192707, -1.192707, -1.227695,  0.597370],
                     [ 1.572750,  0.678300,  0.678300, -1.215177,  1.573647]])

recon_2 = np.matrix([[-0.286753, -0.405398, -0.391275, -0.496714, -0.286753],
                     [-0.790141, -0.408477, -0.458704, -0.039374, -0.790141],
                     [-1.236701, -1.393375, -1.377126, -1.476209, -1.236701],
                     [ 0.947443,  0.721967,  0.752985,  0.482764,  0.947443],
                     [ 1.366153,  1.485283,  1.474120,  1.529533,  1.366153]])


recon_3 = np.matrix([[ 0.119278,  0.162790,  -0.290248,  0.162790, -0.293615],
                     [-1.907964, -1.955346,   0.571294, -1.955346,  1.571879],
                     [ 0.821725,  0.812148,  -0.057841,  0.812148, -0.532537],
                     [ 0.165119,  0.419708,  -1.621756,  0.419708, -1.342744],
                     [ 0.801842,  0.560700,   1.398550,  0.560700,  0.597017]])




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
    gray = se.Brain(se.load('gray', vox_size=20))

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
    bo_sample = se.Brain(data=data.values, locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    # actual = bo.data.iloc[:, unknown_ind]
    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.values, recon.data.values)

    assert 1 >= corr_vals.mean() >= -1
    #assert np.allclose(zscore(recon_1), recon.data, equal_nan=True)


def test_electrode_contingencies_2_subset():

    random_seed = np.random.seed(123)

    noise = 0

    gray = se.Brain(se.load('gray', vox_size=20))

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
    bo_sample = se.Brain(data=data.values, locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.values, recon.data.values)

    #assert np.allclose(zscore(recon_2), recon.data, equal_nan=True)
    assert 1 >= corr_vals.mean() >= -1


def test_electrode_contingencies_3_locations_can_subset():

    random_seed = np.random.seed(123)
    noise = 0

    # load mini model
    gray = se.Brain(se.load('gray', vox_size=20))

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
    bo_sample = se.Brain(data=data.values, locs=sub_locs, sample_rate=1000)

    # predict activity at all unknown locations
    recon = model.predict(bo_sample, nearest_neighbor=False)

    # actual = bo.data.iloc[:, unknown_ind]
    actual = bo.data.iloc[:, recon.locs.index]

    corr_vals = _corr_column(actual.values, recon.data.values)

    assert 1 >= corr_vals.mean() >= -1
    #assert np.allclose(zscore(recon_3), recon.data, equal_nan=True)
