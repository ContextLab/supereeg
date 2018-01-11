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
import sys
import ast
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from superEEG._helpers.stats import r2z, z2r, corr_column
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

from config import config
try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])


# n_samples
n_samples = 1000

# n_electrodes - number of electrodes for reconstructed patient - need to loop over 5:5:130

#n_elecs = range(5, 95, )
#n_elecs = [5,15, 25, 45, 65, 85,  50, 95]
#n_elecs = [ast.literal_eval(n_elecs)]

### debug with input:

#n_elecs = [ast.literal_eval(sys.argv[1])]

# m_patients - number of patients in the model - need to loop over 10:10:50
m_patients = [1, 5, 10]

# m_electrodes - number of electrodes for each patient in the model -  25:25:100
#m_elecs = range(5, 165, 50)
#m_elecs = [100]

iter_val = 1

# load nifti to get locations
gray = se.load('mini_model')

# extract locations
gray_locs = gray.locs

d = []
append_d = pd.DataFrame()
if str(sys.argv [1]) == 'location_case_1':
    param_grid = [(p, m, n) for p in m_patients for m in range(10, 170, 50) for n in range(10, 170 - m, 50)]

if str(sys.argv [1]) == 'location_case_2':
    param_grid = [(p, m, n) for p in m_patients for m in range(10, 170, 50) for n in range(10, m, 50)]

if str(sys.argv [1]) == 'location_case_3':
    param_grid = [(p,m,n) for p in m_patients for m in range(10,170,50) for n in range(10,170,50)]
else:
    print('need to input script paramter to deliniate special location cases')
#for p, m, n in [(10,10,160)]:
for p, m, n in param_grid:
    d = []
    for i in range(iter_val):


############################

# 3 separate contingincies:

############################

        ### 1: no intersection of model locations and brain object locations ( intersection of A and B is null

        if str(sys.argv [1]) == 'location_case_1':
            mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])

            c = se.create_cov(cov='random', n_elecs=170)

            #data = c[:, mo_locs.index][mo_locs.index, :]
            data = c[:, mo_locs.index][mo_locs.index, :]

            model = se.Model(numerator=data*p, denominator=np.ones(np.shape(data))*p, locs=mo_locs, n_subs=p)

            # create brain object from the remaining locations - first find remaining locations
            possible_sub_locs = gray_locs[~gray_locs.index.isin(mo_locs.index)]

            sub_locs = possible_sub_locs.sample(n).sort_values(['x', 'y', 'z'])

            # create a brain object with all gray locations
            bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

            # parse brain object to create synthetic patient data
            data = bo.data.iloc[:, sub_locs.index]

            # put data and locations together in new sample brain object
            bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

            # predict activity at all unknown locations
            recon = model.predict(bo_sample)

            # actual = bo.data.iloc[:, unknown_ind]
            actual = bo.data.iloc[:, recon.locs.index]

            corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

            corr_vals_sample = np.random.choice(corr_vals, 5)

            corr_val_mean = corr_vals_sample.mean()

####################################
        ### 2: all brain object locations are also model locations ( B is a subset of A)

        if str(sys.argv [1]) == 'location_case_2':

            mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])

            c = se.create_cov(cov='random', n_elecs=170)

            data = c[:, mo_locs.index][mo_locs.index, :]

            model = se.Model(numerator=data * p, denominator=np.ones(np.shape(data)) * p, locs=mo_locs, n_subs=p)

            # create brain object from the remaining locations - first find remaining locations
            sub_locs = mo_locs.sample(n).sort_values(['x', 'y', 'z'])

            # create a brain object with all gray locations
            bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

            # get indices for unknown locations (where we wish to predict)
            #unknown_loc = gray_locs[~gray_locs.index.isin(sub_locs.index)]

            # parse brain object to create synthetic patient data
            data = bo.data.iloc[:, sub_locs.index]

            # put data and locations together in new sample brain object
            bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

            print(bo_sample.locs.shape)
            print(model.locs.shape)
            print(gray_locs[~gray_locs.index.isin(sub_locs.index)])
            # predict activity at all unknown locations
            recon = model.predict(bo_sample)

            actual = bo.data.iloc[:, recon.locs.index]

            corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

            corr_vals_sample = np.random.choice(corr_vals, 5)

            corr_val_mean = corr_vals_sample.mean()


############################

        ### 3: some locations in the brain object overlap with the model locations
        if str(sys.argv[1]) == 'location_case_3':
            ### bypassing making the model from brain objects

            mo_locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])

            c = se.create_cov(cov='random', n_elecs=170)

            data = c[:, mo_locs.index][mo_locs.index, :]

            model = se.Model(numerator=data * p, denominator=np.ones(np.shape(data)) * p, locs=mo_locs, n_subs=p)

            # brain object locations subsetted entirely from both model and gray locations - for this n > m (this isn't necessarily true, but this ensures overlap)

            sub_locs = gray_locs.sample(n).sort_values(['x', 'y', 'z'])

            # for the case where you want both subset and disjoint locations - get indices for unknown locations (where we wish to predict)

            #unknown_locs = mo_locs[~mo_locs.index.isin(sub_locs.index)]

            bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

            data = bo.data.iloc[:, sub_locs.index]

            bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs, sample_rate=1000)

            try:
                recon = model.predict(bo_sample)
                # sample actual data at reconstructed locations
                actual = bo.data.iloc[:, recon.locs.index]

                # correlate reconstruction with actual data
                corr_vals = corr_column(actual.as_matrix(), recon.data.as_matrix())

                corr_vals_sample = np.random.choice(corr_vals, 5)
                corr_val_mean = corr_vals_sample.mean()

            except:
                print('m:' + str(m), 'n:' + str(n))
                print('SVD issue')
                corr_vals = float('nan')
                corr_vals_mean = float('nan')


        d.append({'Numbder of Patients in Model': p, 'Number of Model Locations': m, 'Number of Patient Locations': n,
                  'Average Correlation': corr_val_mean, 'Correlations': corr_vals, 'Patient Locations': bo_sample.locs.values})

    append_d = append_d.append(d)
    append_d.index.rename('Iteration', inplace=True)


append_d

if os.path.isfile(os.path.join(config['resultsdir'], 'ave_corrs' + str(sys.argv[1]))):
    f = open(os.path.join(config['resultsdir'], 'ave_corrs' + str(sys.argv[1])), 'a')
    append_d.to_csv(f, mode='a', header=False)
    f.close()
else:
    f = open(os.path.join(config['resultsdir'], 'ave_corrs' + str(sys.argv[1])), 'a')
    append_d.to_csv(f, mode='a', header=True)
    f.close()

new_df=append_d.groupby('Average Correlation').mean()
# new_df['Proportion of electrodes from to-be-reconstructed patient'] = new_df['Number of Model Locations'] / 170
# new_df['Proportion of electrodes from patients used to construct model'] = new_df['Number of Patient Locations'] / 170
if len(np.unique(new_df['Numbder of Patients in Model'])) > 1:

    fig, axs = plt.subplots(ncols=len(np.unique(new_df['Numbder of Patients in Model'])), sharex=True, sharey=True)

    axs_iter = 0
    cbar_ax = fig.add_axes([.92, .3, .03, .4])
    for i in np.unique(new_df['Numbder of Patients in Model']):


        data_plot = append_d[append_d['Numbder of Patients in Model'] == i].pivot_table(index=['Number of Model Locations'], columns='Number of Patient Locations',
                                                              values='Average Correlation')
        axs[axs_iter].set_title('Patients = '+ str(i))
        sns.heatmap(data_plot, cbar = axs_iter == 0, ax = axs[axs_iter], cbar_ax = None if axs_iter else cbar_ax)
        axs[axs_iter].invert_yaxis()
        axs_iter+=1

else:
    for i in np.unique(new_df['Numbder of Patients in Model']):
        data_plot = append_d[append_d['Numbder of Patients in Model'] == i].pivot_table(
            index=['Number of Model Locations'], columns='Number of Patient Locations',
            values='Average Correlation')
        ax = sns.heatmap(data_plot, vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set(xlabel='Number of electrodes from to-be-reconstructed patient', ylabel=' Number of electrodes from patients used to construct model')
        #axs_iter += 1
#
#

#
plt.savefig(os.path.join(config['resultsdir'], str(sys.argv[1]) + '_heatmap.pdf'))

## put in locations of electrodes as well

# sns.jointplot(bo.data.iloc[:, unknown_ind].values.flatten(), predicted)


### if want to retrieve model_bos added to the model:
# model_bos = [se.simulate_bo(n_samples=10000, sample_rate=1000, locs = gray_locs.sample(m).sort_values(['x', 'y', 'z'])) for x in range(p)]

# model_locs = pd.DataFrame()
# for i in range(len(model_bos)):
#     #locats = model_bos[i].locs
#     model_locs = model_locs.append(model_bos[i].locs, ignore_index = True)
### and at the end:
# d.append({'Numbder of Patients in Model': p, 'Number of Model Locations': m, 'Number of Patient Locations': n, 'Average Correlation': corr_vals_sample.mean(), 'Correlations': corr_vals, 'Model Locations': model_locs.values, 'Patient Locations': bo_sample.locs.values})
