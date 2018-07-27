
# -*- coding: utf-8 -*-
"""
=============================
debug predict
=============================

This example shows debugging process for predict.  Delete before pip push.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT


import supereeg as se
import sys
import numpy as np
from supereeg.helpers import _corr_column, _count_overlapping
try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

from scipy.stats import zscore
#
# def round_it(locs, places):
#     """
#     Rounding function
#
#     Parameters
#     ----------
#     locs : float
#         Number be rounded
#
#     places : int
#         Number of places to round
#
#     Returns
#     ----------
#     result : float
#         Rounded number
#
#
#     """
#     return np.round(locs, decimals=places)
#
# def get_rows(all_locations, subj_locations):
#     """
#         This function indexes a subject's electrode locations in the full array of electrode locations
#
#         Parameters
#         ----------
#         all_locations : ndarray
#             Full array of electrode locations
#
#         subj_locations : ndarray
#             Array of subject's electrode locations
#
#         Returns
#         ----------
#         results : list
#             Indexs for subject electrodes in the full array of electrodes
#
#         """
#     if subj_locations.ndim == 1:
#         subj_locations = subj_locations.reshape(1, 3)
#     inds = np.full([1, subj_locations.shape[0]], np.nan)
#     for i in range(subj_locations.shape[0]):
#         possible_locations = np.ones([all_locations.shape[0], 1])
#         try:
#             for c in range(all_locations.shape[1]):
#                 possible_locations[all_locations[:, c] != subj_locations[i, c], :] = 0
#             inds[0, i] = np.where(possible_locations == 1)[0][0]
#         except:
#             pass
#     inds = inds[~np.isnan(inds)]
#     return [int(x) for x in inds]
#
# def known_unknown(fullarray, knownarray, subarray=None, electrode=None):
#     """
#         This finds the indices for known and unknown electrodes in the full array of electrode locations
#
#         Parameters
#         ----------
#         fullarray : ndarray
#             Full array of electrode locations - All electrodes that pass the kurtosis test
#
#         knownarray : ndarray
#             Subset of known electrode locations  - Subject's electrode locations that pass the kurtosis test (in the leave one out case, this is also has the specified location missing)
#
#         subarray : ndarray
#             Subject's electrode locations (all)
#
#         electrode : str
#             Index of electrode in subarray to remove (in the leave one out case)
#
#         Returns
#         ----------
#         known_inds : list
#             List of known indices
#
#         unknown_inds : list
#             List of unknown indices
#
#         """
#     ## where known electrodes are located in full matrix
#     known_inds = get_rows(round_it(fullarray, 3), round_it(knownarray, 3))
#     ## where the rest of the electrodes are located
#     unknown_inds = list(set(range(np.shape(fullarray)[0])) - set(known_inds))
#     if not electrode is None:
#         ## where the removed electrode is located in full matrix
#         rm_full_ind = get_rows(round_it(fullarray, 3), round_it(subarray[int(electrode)], 3))
#         ## where the removed electrode is located in the unknown index subset
#         rm_unknown_ind = np.where(np.array(unknown_inds) == np.array(rm_full_ind))[0].tolist()
#         return known_inds, unknown_inds, rm_unknown_ind
#     else:
#         return known_inds, unknown_inds
#
#
# def chunker(iterable, n, fillvalue=None):
#     #"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
#     args = [iter(iterable)] * n
#     return zip_longest(fillvalue=fillvalue, *args)
#
# def time_by_file_index_bo(bo, ave_data, known_inds, unknown_inds):
#     """
#     Session dependent function that calculates that finds either the timeseries or the correlation of the predicted and actual timeseries for a given location chunked by 25000 timepoints
#
#     Parameters
#     ----------
#     fname : Data matrix (npz file)
#         The data to be analyzed.
#         Filename containing fields:
#             Y - time series
#             R - electrode locations
#             fname_labels - session number
#             sample_rate - sampling rate
#
#     ave_data: ndarray
#         Average correlation matrix
#
#     known_inds: list
#         Indices for known electrodes in average matrix
#
#     unknown_inds: list
#         Indices for unknown electrodes in average matrix
#
#     electrode_ind: int
#         Index for estimated location in average matrix (location in unknown_inds)
#
#     k_flat_removed: list
#         Indices of good channels (pass kurtosis test) in Y
#
#     electrode: int
#         Index of held out location in known_inds
#
#     time_series: boolean
#         True: output is predicted and actual timeseries
#         False: output is predicted and actual correlation
#
#     Returns
#     ----------
#     results : pandas dataframe
#         If timeseries input is:
#         True: output is predicted and actual timeseries
#         False: output is predicted and actual correlation
#
#
#     """
#     file_inds = np.unique(np.atleast_2d(bo.sessions.as_matrix()))
#     Kaa = np.float32(ave_data[known_inds, :][:, known_inds])
#     Kaa_inv = np.linalg.pinv(Kaa)
#     Kba = np.float32(ave_data[unknown_inds, :][:, known_inds])
#     results = []
#     for i in file_inds:
#         if np.shape(np.atleast_2d(bo.sessions.as_matrix()))[1] == 1:
#             fname_labels = np.atleast_2d(bo.sessions.as_matrix()).T
#         else:
#             fname_labels = np.atleast_2d(bo.sessions.as_matrix())
#         next_inds = np.where(fname_labels == i)[1]
#         ### this code should incorporate the average voltage of the known (subject) electrodes and the average for the unknown (the other subjects)
#         block_results = []
#         next = np.zeros((bo.get_data().shape[0], ave_data.shape[0]))
#         ### right now, this doesn't use an overlap in time, but this needs to be addressed when I see edge effects
#         for each in chunker(next_inds, 1000):
#
#             next[:, unknown_inds] = np.squeeze(np.dot(np.dot(Kba, Kaa_inv),
#                                                zscore(np.float32(
#                                                    bo.get_data().as_matrix()[filter(lambda v: v is not None, each), :])).T).T)
#             next[:, known_inds] = np.squeeze(zscore(np.float32(bo.get_data().as_matrix()[filter(lambda v: v is not None, each), :])))
#             if block_results==[]:
#                 block_results = next
#             else:
#                 block_results = np.vstack((block_results, next))
#         if results==[]:
#             results = block_results
#         else:
#             results = np.vstack((block_results, results))
#
#         return results

#
# # simulate 100 locations
# locs = se.simulate_locations(n_elecs=100, random_seed=True)
#
# # simulate brain object
# bo = se.simulate_bo(n_samples=1000, sample_rate=100, cov='random', locs=locs, noise=0, random_seed=True)
#
# # sample 10 locations, and get indices
# sub_locs = locs.sample(90, replace=False, random_state=123).sort_values(['x', 'y', 'z']).index.values.tolist()
#
# # index brain object to get sample patient
# bo_sample = bo[: ,sub_locs]
#
# # plot sample patient locations
# bo_sample.plot_locs()
#
# # plot sample patient data
# bo_sample.plot_data()
#
# Model = se.Model(data=bo, locs=locs)
#
# R = Model.get_locs().as_matrix()
#
# R_K_subj = bo_sample.get_locs().as_matrix()
#
# known_inds, unknown_inds = known_unknown(R, R_K_subj, R_K_subj)
#
#
#
# recon_data = time_by_file_index_bo(bo_sample, Model.get_model(z_transform=False), known_inds, unknown_inds)
#
# bo_r = se.Brain(data=recon_data, locs = R, sample_rate=bo.sample_rate, sessions=bo.sessions.as_matrix())
#
#
# corrs_1 = _corr_column(bo.get_data().as_matrix(), bo_r.get_data().as_matrix())
#
# print('correlations with timeseries recon  = ' + str(corrs_1[unknown_inds].mean()))
#
#
# bo_s = Model.predict(bo_sample, nearest_neighbor=False)
#
# recon_labels = np.where(np.array(bo_s.label) != 'observed')
#
# corrs = _corr_column(bo.get_data().as_matrix(), bo_s.get_data().as_matrix())
#
# print('correlations with predict function = ' + str(corrs[recon_labels].mean()))
#
# assert np.allclose(corrs, corrs_1)


########## debug case 1 - null set ##################

# set random seed to default and noise to 0
random_seed = np.random.seed(123)
noise = 0

# locs
locs = se.simulate_locations(n_elecs=100, set_random_seed=random_seed)

# create model locs from 75 locations
mo_locs = locs.sample(75, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create covariance matrix from random seed
c = se.create_cov(cov='random', n_elecs=100)

# pull out model from covariance matrix
data = c[:, mo_locs.index][mo_locs.index, :]

# create model from subsetted covariance matrix and locations
model = se.Model(data=data, locs=mo_locs, n_subs=1)

# create brain object from the remaining locations - first find remaining 25 locations
sub_locs = locs[~locs.index.isin(mo_locs.index)]

# create a brain object with all gray locations
bo = se.simulate_bo(n_samples=1000, sample_rate=100, locs=locs, noise=noise, random_seed=random_seed)

# parse brain object to create synthetic patient data
data = bo.data.iloc[:, sub_locs.index]

# put data and locations together in new sample brain object
bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=100)

# predict activity at all unknown locations
recon = model.predict(bo_sample, nearest_neighbor=False)

# get reconstructed indices
recon_labels = np.where(np.array(recon.label) != 'observed')

# actual = bo.data.iloc[:, unknown_ind]
actual_data = bo.get_zscore_data()[:, recon_labels[0]]

recon_data = recon[:, recon_labels[0]].get_data().as_matrix()
corr_vals = _corr_column(actual_data, recon_data)

print('case 1 (null set) correlation = ' +str(corr_vals.mean()))




########## debug case 2 - subset ##################

# set random seed to default and noise to 0
random_seed = np.random.seed(123)
noise = 0

# locs
locs = se.simulate_locations(n_elecs=100, set_random_seed=random_seed)

# create model locs from 50 locations
mo_locs = locs.sample(100, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create covariance matrix from random seed
c = se.create_cov(cov='random', n_elecs=100)

# pull out model from covariance matrix
data = c[:, mo_locs.index][mo_locs.index, :]

# create model from subsetted covariance matrix and locations
model = se.Model(data=data, locs=mo_locs, n_subs=1)

# create brain object from subset of model locations
sub_locs = mo_locs.sample(25, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create a brain object with all gray locations
bo = se.simulate_bo(n_samples=1000, sample_rate=100, locs=mo_locs, noise=noise, random_seed=random_seed)

# parse brain object to create synthetic patient data
data = bo.data.iloc[:, sub_locs.index]

# put data and locations together in new sample brain object
bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=100)

# predict activity at all unknown locations
recon = model.predict(bo_sample, nearest_neighbor=False)

# get reconstructed indices
recon_labels = np.where(np.array(recon.label) != 'observed')

# actual = bo.data.iloc[:, unknown_ind]
actual_data = bo.get_zscore_data()[:, recon_labels[0]]

recon_data = recon[:, recon_labels[0]].get_data().as_matrix()
corr_vals = _corr_column(actual_data, recon_data)

print('case 2 (subset of model) correlation = ' +str(corr_vals.mean()))

########## debug case 3 - overlapping set ##################

# set random seed to default and noise to 0
random_seed = np.random.seed(123)
noise = 0

# locs
locs = se.simulate_locations(n_elecs=100, set_random_seed=random_seed)

# create model locs from 75 locations
mo_locs = locs.sample(75, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create covariance matrix from random seed
c = se.create_cov(cov='random', n_elecs=100)

# pull out model from covariance matrix
data = c[:, mo_locs.index][mo_locs.index, :]

# create model from subsetted covariance matrix and locations
model = se.Model(data=data, locs=mo_locs, n_subs=1)

# create brain object from all the locations - first find remaining 25 location
sub_locs = locs[~locs.index.isin(mo_locs.index)]

# then add 25 locations subsetted from model locations
sub_locs = sub_locs.append(mo_locs.sample(25, random_state=random_seed).sort_values(['x', 'y', 'z']))

# then subsample 25 from those locations to get some overlapping
sub_locs.sample(25, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create a brain object with all gray locations
bo = se.simulate_bo(n_samples=1000, sample_rate=100, locs=locs, noise=noise, random_seed=random_seed)

# parse brain object to create synthetic patient data
data = bo.data.iloc[:, sub_locs.index]

# put data and locations together in new sample brain object
bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=100)

# predict activity at all unknown locations
recon = model.predict(bo_sample, nearest_neighbor=False)

# get reconstructed indices
recon_labels = np.where(np.array(recon.label) != 'observed')

# actual = bo.data.iloc[:, unknown_ind]
actual_data = bo.get_zscore_data()[:, recon_labels[0]]

recon_data = recon[:, recon_labels[0]].get_data().as_matrix()
corr_vals = _corr_column(actual_data, recon_data)

print('case 3 (some overlap of model) correlation = ' +str(corr_vals.mean()))

########## debug case 4 - model subset of brain object ##################

# set random seed to default and noise to 0
random_seed = np.random.seed(123)
noise = 0

# locs
locs = se.simulate_locations(n_elecs=100, set_random_seed=random_seed)

# create brain locs from 75 locations
bo_locs = locs.sample(75, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create model locs from 50 locations
mo_locs = bo_locs.sample(50, random_state=random_seed).sort_values(['x', 'y', 'z'])

# create covariance matrix from random seed
c = se.create_cov(cov='random', n_elecs=100)

# pull out model from covariance matrix
data = c[:, mo_locs.index][mo_locs.index, :]

# create model from subsetted covariance matrix and locations
model = se.Model(data=data, locs=mo_locs, n_subs=1)


# create a brain object with all gray locations
bo = se.simulate_bo(n_samples=1000, sample_rate=100, locs=locs, noise=noise, random_seed=random_seed)

# parse brain object to create synthetic patient data
data = bo.data.iloc[:, bo_locs.index]

# put data and locations together in new sample brain object
bo_sample = se.Brain(data=data.as_matrix(), locs=bo_locs, sample_rate=100)

# predict activity at all unknown locations
recon = model.predict(bo_sample, nearest_neighbor=False)

# get reconstructed indices - since model is entirely a subset of brain object,
# there should be no reconstructed locations
recon_labels = np.where(np.array(recon.label) != 'observed')

# actual = bo.data.iloc[:, unknown_ind]
actual_data = bo_sample.get_zscore_data()

recon_data = recon.get_data().as_matrix()
corr_vals = _corr_column(actual_data, recon_data)

print('case 4 (model subset of brain locs) correlation = ' +str(corr_vals.mean()))