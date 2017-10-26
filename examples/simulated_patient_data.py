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
import matplotlib.pyplot as plt

# n_samples
n_samples = 1000

# n_electrodes - number of electrodes for reconstructed patient - need to loop over 5:5:130
#n_elecs = range(5,160,50)
n_elecs =[100, 50, 10]
#n_elecs = [165]

# m_patients - number of patients in the model - need to loop over 10:10:50
m_patients = [50, 10, 5]

# m_electrodes - number of electrodes for each patient in the model -  25:25:100
#m_elecs = range(5,160,50)
m_elecs = [150, 50, 5]

iter_val = 5

# load nifti to get locations
gray = se.load(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_20mm_brain.nii')

# extract locations
gray_locs = gray.locs

d = []
append_d = pd.DataFrame()

param_grid = [(p,m,n) for p in m_patients for m in m_elecs for n in n_elecs]

for p, m, n in param_grid:
    d = []

    for i in range(iter_val):


############################

# 3 separate contingincies:

############################

       #  ### 1: no intersection of model locations and brain object locations ( intersection of A and B is null )
       #
       #  # subset locations to build model
       #  mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])
       #
       #  #create brain objects with m_patients and loop over the number of model locations
       #  model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = mo_locs) for x in range(p)]
       #
       #  # create model from subsampled gray locations
       #  model = se.Model(model_bos, locs=mo_locs)
       #
       #  # create brain object from the remaining locations - first find remaining locations
       #  sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]
       #
       #  # create a brain object with all gray locations
       #  bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)
       #
       #  # get indices for unknown locations (where we wish to predict)
       #  unknown_loc = mo_locs[~mo_locs.index.isin(sub_locs.index)]
       #
       #  # parse brain object to create synthetic patient data
       #  data = bo.data.T.drop(unknown_loc.index).T
       #
       #  # put data and locations together in new sample brain object
       #  bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)
       #
       #  # predict activity at all unknown locations
       #  recon = model.predict(bo_sample)
       #
       #  # this next step is redundant - just get from unknown_loc later
       # # unknown_ind = [item for item in bo.data.columns if item not in data.columns]
       #
       #  #actual = bo.data.iloc[:, unknown_ind]
       #  actual = bo.data.iloc[:, recon.locs.index]
       #
       #  corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

####################################

        # ### 2: all brain object locations are also model locations ( B is a subset of A)
        #
        # # subset gray locations to build model
        # mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])
        #
        # #create brain objects with m_patients and loop over the number of model locations
        # model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = mo_locs) for x in range(p)]
        #
        # # create model from subsampled
        # model = se.Model(model_bos, locs=mo_locs)
        #
        # # brain object locations subsetted entirely from model locations - for this m > n
        # sub_locs = mo_locs.sample(n).sort_values(['x', 'y', 'z'])
        #
        # # create a brain object with all gray locations
        # bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)
        #
        # # get indices for unknown locations (where we wish to predict) indices for gray_locs - sub_locs
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
        # # sample actual data at reconstructed locations
        # actual = bo.data.iloc[:, recon.locs.index]
        #
        # corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

############################

        ### 3: some locations in the brain object overlap with the model locations

        # subset locations to build model
        #mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])

        #create brain objects with m_patients and loop over the number of model locations
        model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])) for x in range(p)]

        # create model from subsampled gray locations
        model = se.Model(model_bos, locs=gray_locs)

        # # brain object locations subsetted entirely from both model and gray locations - for this n > m (this isn't necessarily true, but this ensures overlap)
        sub_locs = gray_locs.sample(n).sort_values(['x', 'y', 'z'])

        #### add while loop that samples from both modle and gray locations

        # for the case where you want both subset and disjoint locations - get indices for unknown locations (where we wish to predict)
        unknown_loc = gray_locs[~gray_locs.index.isin(sub_locs.index)]

        bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

        data = bo.data.T.drop(unknown_loc.index).T
        bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)

        recon = model.predict(bo_sample)

        # sample actual data at reconstructed locations
        actual = bo.data.iloc[:, recon.locs.index]

        #correlate reconstruction with actual data
        corr_vals = corr_column(actual.as_matrix(),recon.data.as_matrix())
        #
        # # sns.jointplot(bo.data.iloc[:, unknown_ind].values.flatten(), predicted)

        d.append({'Patients': p, 'Model Locations': m, 'Patient Locations': n, 'Ave_corr': corr_vals.mean(), 'Correlations': corr_vals})

    d = pd.DataFrame(d)
    append_d = append_d.append(d)

append_d


new_df=append_d.groupby('Ave_corr').mean()


fig, axs = plt.subplots(ncols=2)
axs_iter = 0
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i in np.unique(new_df['Patients']):

    data_plot = append_d[append_d.Patients == i].pivot_table(index=['Model Locations'], columns='Patient Locations',
                                                          values='Ave_corr')
    axs[axs_iter].set_title('Patients = '+ str(i))
    bar_ax = None if axs_iter else cbar_ax
    sns.heatmap(data_plot, annot=True, fmt=".1f", ax = axs[axs_iter], cbar=cbar_ax)
    axs_iter+=1


plt.tight_layout()
plt.savefig('Average_correlation_heatmap.png')

