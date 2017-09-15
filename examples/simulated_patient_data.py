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
from superEEG._helpers.stats import r2z, z2r, corr_column


# n_samples
n_samples = 1000

# n_electrodes - number of electrodes for reconstructed patient - need to loop over 5:5:130
n_elecs =[169, 50]
#n_elecs = [165]

# m_patients - number of patients in the model - need to loop over 10:10:50
m_patients = [10, 50]

# m_electrodes - number of electrodes for each patient in the model -  25:25:100
m_elecs = [170, 10]

# load nifti to get locations
gray = se.load(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_20mm_brain.nii')

# extract locations
gray_locs = gray.locs


d = []

param_grid = [(p,m,n) for p in m_patients for m in m_elecs for n in n_elecs]

for p, m, n in param_grid:
    sub_locs = gray_locs.sample(m)
    #
    #predict_loc = locs.drop(p_n_elecs)
    #create brain objects with m_patients and loop over the number of model locations
    model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = sub_locs) for i in range(p)]

    model = se.Model(model_bos)
    model.plot(yticklabels=False, xticklabels=False)
    bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=model.locs)
    data = bo.data.T.sample(n).T
    bo_sample = se.Brain(data=data.as_matrix(), locs=bo.locs.loc[data.columns, :].as_matrix())
    recon = model.predict(bo_sample)
    unknown_ind = [item for item in bo.data.columns if item not in data.columns]

    # predicted = recon.data.iloc[:, unknown_ind]
    # actual = bo.data.iloc[:, unknown_ind]

    corr_vals = corr_column(np.atleast_2d(bo.data.iloc[:, unknown_ind].as_matrix()), np.atleast_2d(recon.data.iloc[:, unknown_ind].as_matrix()))

    d.append({'Patients': p, 'Model Locations': m, 'Patient Locations': n, 'Correlation': corr_vals})

d = pd.DataFrame(d)













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