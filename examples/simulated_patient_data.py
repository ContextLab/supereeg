# -*- coding: utf-8 -*-
"""
=============================
Simulate data
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se
import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from scipy.stats import kurtosis, zscore, pearsonr
from scipy.spatial.distance import cdist
from superEEG._helpers.stats import r2z, z2r, corr_column

def recon(bo_sub, R):
    """
    """
    R[np.eye(R.shape[0]) == 1] = 1
    known_inds = bo_sub.locs.index.values
    locs_inds = range(R.shape[0])
    unknown_inds = np.sort(list(set(locs_inds) - set(known_inds)))
    Kba = R[unknown_inds, :][:, known_inds]
    Kaa = R[:,known_inds][known_inds,:]
    Y = zscore(bo_sub.get_data())
    return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)


def recon_m(bo_sub, mo):
    """
    """
    model = z2r(np.divide(mo.numerator, mo.denominator))
    model[np.eye(model.shape[0]) == 1] = 1
    known_locs = bo_sub.locs
    known_inds = bo_sub.locs.index.values
    unknown_locs = mo.locs.drop(known_inds)
    unknown_inds = unknown_locs.index.values
    Kba = model[unknown_inds, :][:, known_inds]
    Kaa = model[:,known_inds][known_inds,:]
    Y = zscore(bo_sub.get_data())
    return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)

def row_in_array(myarray, myrow):
    """
        Looks to see if a row (electrode location) is in the bigger array

        Parameters
        ----------
        myarray : ndarray
            Larger array of electrode locations

        myrow : ndarray
            Specific row to find

        Returns
        ----------
        results : bool
            True if row in array; False if not

        """
    return (myarray == myrow).all(-1).any()

# n_samples
n_samples = 1000

# n_electrodes - number of electrodes for reconstructed patient - need to loop over 5:5:130
n_elecs =[70, 50, 5]
#n_elecs = [165]

# m_patients - number of patients in the model - need to loop over 10:10:50
m_patients = [10, 50]

# m_electrodes - number of electrodes for each patient in the model -  25:25:100
m_elecs = [100, 50, 5]

iter_val = 10

# load nifti to get locations
gray = se.load(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_20mm_brain.nii')

# extract locations
gray_locs = gray.locs

d = []

R = 1 - cdist(gray_locs, gray_locs, metric='euclidean')
R -= np.min(R)
R /= np.max(R)
R = 2 * R - 1


param_grid = [(p,m,n) for p in m_patients for m in m_elecs for n in n_elecs]

for p, m, n in param_grid:

    for i in range(iter_val):


        ### 3 separate contingincies:

        # if possible - make these fail if n > m or m > n


####################

        # ### 1: no intersection of model locations and brain object locations ( intersection of A and B is null )
        #
        # # subset locations to build model
        # mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])
        #
        # #create brain objects with m_patients and loop over the number of model locations
        # model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = mo_locs) for x in range(p)]
        #
        # # create model from subsampled gray locations
        # model = se.Model(model_bos, locs=mo_locs)
        #
        # # create brain object from the remaining locations - first find remaining locations
        # sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]
        #
        # # create a brain object with all gray locations
        # bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)
        #
        # # get indices for unknown locations (where we wish to predict)
        # unknown_loc = mo_locs[~mo_locs.index.isin(sub_locs.index)]
        #
        # # parse brain object to create synthetic patient data
        # data = bo.data.T.drop(unknown_loc.index).T
        #
        # # put data and locations together in new sample brain object
        # bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)
        #
        # # predict activity at all unknown locations
        # recon = model.predict(bo_sample)
        #
        # # this next step is redundant - just get from unknown_loc later
        # unknown_ind = [item for item in bo.data.columns if item not in data.columns]
        #
        # actual = bo.data.iloc[:, unknown_ind]
        #
        # corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

####################################

        # ### 2: all brain object locations are also model locations ( B is a subset of A)
        #
        # # subset locations to build model
        # mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])
        #
        # #create brain objects with m_patients and loop over the number of model locations
        # model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = mo_locs) for x in range(p)]
        #
        # # create model from subsampled gray locations
        # model = se.Model(model_bos, locs=gray_locs)
        #
        # # # model.plot(yticklabels=False, xticklabels=False)
        #
        # # brain object locations subsetted entirely from model - for this m > n
        # sub_locs = gray_locs.sample(n).sort_values(['x', 'y', 'z'])
        #
        # # create a brain object with all gray locations
        # bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)
        #
        # # get indices for unknown locations (where we wish to predict)
        # unknown_loc = gray_locs[~gray_locs.index.isin(sub_locs.index)]
        #
        # # parse brain object to create synthetic patient data
        # data = bo.data.T.drop(unknown_loc.index).T
        #
        # # put data and locations together in new sample brain object
        # bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)
        #
        # # predict activity at all unknown locations
        # recon = model.predict(bo_sample)
        #
        # # get indices for unknown locations (where we wish to predict)
        # unknown_ix = mo_locs[~mo_locs.index.isin(sub_locs.index)]
        #
        # # this next step is redundant - just get from unknown_loc later
        # #unknown_ind = [item for item in bo.data.columns if item not in data.columns]
        #
        # actual = bo.data.iloc[:, unknown_ix.index]
        #
        # corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

############################

        ### 3: some locations in the brain object overlap with the model locations

        # subset locations to build model
        mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])

        #create brain objects with m_patients and loop over the number of model locations
        model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = mo_locs) for x in range(p)]

        # create model from subsampled gray locations
        model = se.Model(model_bos, locs=mo_locs)

        # # brain object locations subsetted entirely from model and gray locations - for this n > m
        sub_locs = gray_locs.sample(n).sort_values(['x', 'y', 'z'])


        # for the case where you want both subset and disjoint locations - get indices for unknown locations (where we wish to predict)
        unknown_loc = gray_locs[~gray_locs.index.isin(sub_locs.index)]

        bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

        data = bo.data.T.drop(unknown_loc.index).T
        bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)

        recon = model.predict(bo_sample)
        #
        # # predicted = pd.DataFrame(recon(bo_sample, R))
        #
        # #predicted_m = pd.DataFrame(recon_m(bo_sample, model))
        #
        unknown_ind = [item for item in bo.data.columns if item not in data.columns]
        #
        # # predicted = recon.data.iloc[:, unknown_ind]
        # # actual = bo.data.iloc[:, unknown_ind]
        #
        actual = bo.data.iloc[:, unknown_ind]
        #
        corr_vals = corr_column(actual.as_matrix(),recon.data.as_matrix())
        #
        # # sns.jointplot(bo.data.iloc[:, unknown_ind].values.flatten(), predicted)

        d.append({'Patients': p, 'Model Locations': m, 'Patient Locations': n, 'Correlation': corr_vals})

    d = pd.DataFrame(d)

    iter_average = iter_average.append(d[['Patients', 'Model Locations', 'Patient Locations', 'Correlation']].mean(axis=0))


iter_average








# import superEEG as se
# # small model
# model = se.load('example_model')
# data = se.load('example_data')


# #### starting to predict parallelization
# reconstruct = model.predict(data)


# # create directory for synthetic patient data
# synth_dir = os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/synthetic_data'
# if not os.path.isdir(synth_dir):
#     os.mkdir(synth_dir)
#
# # create 50 synthetic patients data with activity at every location
# if not os.listdir(synth_dir):
#
#     ### for toeplitz matrix
#     # R = scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1])
#
#     ### for distance matrix
#     D = squareform(pdist(locs))
#     R = np.max(D) - D
#     R = R - np.min(R)
#     R = R / np.max(R)
#
#     count = 0
#     for ps in range(50):
#         rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)
#         bo = se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(R)), locs=pd.DataFrame(locs, columns=['x', 'y', 'z']))
#         bo.save(os.path.join(synth_dir, 'synthetic_'+ str(count).rjust(2, '0')))
#         count += 1
# else:
#     print os.listdir(synth_dir)
#
#
# # initiate model
# model_data = []
#
# # initiate dataframe
# d = []
#
# for p in m_patients:
#
#     # random sample m_patients from 50 simulated patients - this will also need to be in a loop
#     patients = np.random.choice(range(50), p, replace=False)
#
#     # loop over n_elecs
#     for n in n_elecs:
#
#         # hold out one patient at a time
#         for i in patients:
#
#             # random sample n locations from 170 locations
#             p_n_elecs = np.sort(np.random.choice(range(len(locs)), n, replace=False))
#
#             ### to debug expand_corrmat:
#             # p_n_elecs = range(10,15)
#
#             with open(os.path.join(synth_dir, 'synthetic_'+ str(i).rjust(2, '0') + '.bo'), 'rb') as handle:
#                 bo_actual = pickle.load(handle)
#                 bo_sub = se.Brain(data=bo_actual.data.loc[:, p_n_elecs],locs= bo_actual.locs.loc[p_n_elecs])
#
#
#             unknown_locs = locs.drop(p_n_elecs)
#             unknown_inds = unknown_locs.index.values
#
#             ##### create model from every other patient
#             # model_patients = [p for p in patients if p != i]
#             # for mp in model_patients:
#             #
#             #     # random sample m_elecs locations from 170 locations (this will also need to be looped over for coverage simulation)
#             #     p_m_elecs = np.sort(np.random.choice(range(len(locs)), m_elecs, replace=False))
#             #
#             #     with open(os.path.join(synth_dir, 'synthetic_' + str(mp).rjust(2, '0') + '.bo'), 'rb') as handle:
#             #         bo = pickle.load(handle)
#             #         model_data.append(se.Brain(data=bo.data.loc[:, p_m_elecs], locs=bo.locs.loc[p_m_elecs]))
#             #
#             # model = se.Model(data=model_data, locs=locs)
#
#             ### to use simulated model
#             with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/model_170.mo', 'rb') as a:
#                 model = pickle.load(a)
#
#
#             # #### comparing second corrmat_expand
#             # #### expand all
#             # reconstructed_predict = model.predict(bo_sub)
#             # #### only expand into unknownxknown and knownxknown
#             # reconstructed_fit = model.predict(bo_sub, prediction=True)
#             # #### check if they give the same values
#             # corr_reconstructions = np.mean(corr_column(reconstructed_predict.data.as_matrix(), reconstructed_fit.data.as_matrix()))
#             # #### comparing second corrmat_expand
#
#             reconstructed_predict = model.predict(bo_sub)
#
#             ##### to use predict function (averaging the subject's expanded matrix with the model) but bypass the second expanded
#             reconstructed = model.predict(bo_sub, simulation=True)
#             predicted = reconstructed.data.as_matrix()
#
#             corr_reconstructions = np.mean(corr_column(reconstructed_predict.data.as_matrix(), predicted))
#             ##### to bypass predict function entirely (and only parse model):
#             # predicted = recon_no_expand(bo_sub, model)
#
#             actual = zscore(bo_actual.data.loc[:, unknown_inds].as_matrix())
#
#             corr_vals = corr_column(actual, predicted)
#
#
#             d.append({'Patients': p, 'Model Locations': m_elecs, 'Patient Locations': n, 'Correlation': np.mean(corr_vals)})
#
# d = pd.DataFrame(d)
#



###### to debug the expand_corrmat mode ='predict'




# ### reconstruction with model (no expanding)
# for n in n_elecs:
#
#     ## to create a model from all 50 simulated patients
#     for i in patients:
#
#         p_n_elecs = np.sort(np.random.choice(range(len(locs)), n, replace=False))
#
#         with open(os.path.join(synth_dir, 'synthetic_'+ str(i).rjust(2, '0') + '.bo'), 'rb') as handle:
#             bo_actual = pickle.load(handle)
#             bo_sub = se.Brain(data=bo_actual.data.loc[:, p_n_elecs],locs= bo_actual.locs.loc[p_n_elecs])
#
#         if not os.path.isfile('model_170.mo'):
#             # create model from every patient
#             model_patients = [p for p in patients]
#             for m in model_patients:
#
#                 with open(os.path.join(synth_dir, 'synthetic_' + str(m).rjust(2, '0') + '.bo'), 'rb') as handle:
#                     bo = pickle.load(handle)
#                     #model_data.append(se.Brain(data=bo.data.loc[:, unknown_inds], locs=bo.locs.loc[unknown_inds]))
#                     model_data.append(se.Brain(data=bo.data, locs=bo.locs))
#
#             model = se.Model(data=model_data, locs=locs)
#
#             with open('model_170.mo', 'wb') as h:
#                 pickle.dump(model, h)
#
#         with open('model_170.mo', 'rb') as a:
#             mo = pickle.load(a)
#
#         def recon_no_expand(bo_sub, mo):
#             """
#             """
#             model = z2r(np.divide(mo.numerator, mo.denominator))
#             model[np.eye(model.shape[0]) == 1] = 1
#             known_locs = bo_sub.locs
#             known_inds = bo_sub.locs.index.values
#             unknown_locs = mo.locs.drop(known_inds)
#             unknown_inds = unknown_locs.index.values
#             Kba = model[unknown_inds, :][:, known_inds]
#             Kaa = model[:,known_inds][known_inds,:]
#             Y = zscore(bo_sub.get_data())
#             return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)
#
#
#         def corr(X, Y):
#             return np.array([scipy.stats.pearsonr(x, y)[0] for x, y in zip(X.T, Y.T)])
#
#
#         predicted = np.atleast_2d(recon_no_expand(bo_sub, mo))
#
#         unknown_locs = locs.drop(p_n_elecs)
#         unknown_inds = unknown_locs.index.values
#
#         actual = zscore(bo_actual.data.loc[:, unknown_inds].as_matrix())
#
#         corr_vals = corr(actual, predicted)
#
#
#         d.append({'Patients': m_patients, 'Model Locations': m_elecs, 'Patient Locations': n, 'Correlation': np.mean(corr_vals)})
#
# d = pd.DataFrame(d)
#
# recon_corr = d.groupby(['Patient Locations']).mean()
#






### create model with synthetic patient data, random sample 20 electrodes
#
# R = scipy.linalg.toeplitz(np.linspace(0,1,len(locs))[::-1])
# data = []
# for i in range(50):
#
#     p = np.random.choice(range(len(locs)), 20, replace=False)
#
#     rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)
#     data.append(se.Brain(data=np.dot(rand_dist, scipy.linalg.cholesky(R))[:,p], locs=pd.DataFrame(locs[p,:], columns=['x', 'y', 'z'])))
#
#     #bo.to_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/synthetic_' + str(i))
# model = se.Model(data=data, locs=locs)

# ## create brain object to be reconstructed
# ## find indices
# locs_inds = range(0,len(locs))
# sub_inds = np.sort(np.random.choice(range(len(locs)), 20, replace=False))
# unknown_inds = list(set(locs_inds)-set(sub_inds))
#
# rand_dist = np.random.multivariate_normal(np.zeros(len(locs)), np.eye(len(locs)), size=n_samples)
# full_data = np.dot(rand_dist, scipy.linalg.cholesky(R))
# bo_sub = se.Brain(data=full_data[:, sub_inds], locs=pd.DataFrame(locs[sub_inds, :], columns=['x', 'y', 'z']))
# bo_actual = se.Brain(data=full_data, locs=pd.DataFrame(locs, columns=['x', 'y', 'z']))
#
# ## need to figure out if we want to keep the activty used to predict - right now its added at the end, but that might not be the best
# reconstructed = model.predict(bo_sub)
#
# import seaborn as sb
# sb.jointplot(reconstructed.data, bo_actual.data)
