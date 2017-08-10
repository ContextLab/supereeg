#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat_parsed, expand_matrix, r2z, z2r, compute_coord, expand_corrmat_j
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, partition_jobs,  known_unknown
import sys
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial.distance import squareform as squareform
import pandas as pd
from scipy.stats import zscore


## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, r, k_thresh):

    loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')

    ## check if cor_fig and full directories exist

    check_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'check')
    if not os.path.isdir(check_dir):
        os.mkdir(check_dir)


    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(check_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npy')):

        ## load subject's electrodes
        npz_data = np.load(fname, mmap_mode='r')
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        ## load average matrix
        Ave_data = np.load(
            os.path.join(average_dir, 'average_model' + '_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'),
            mmap_mode='r')
        R_subj = sub_data['R_subj'] # electrode locations
        C_subj = sub_data['C_subj'] # subject data
        K_subj = sub_data['K_subj'] # kurtosis - 1 by n_elecs
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
        # index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj, k_flat= good_chans(K_subj, R_subj, k_thresh, C = C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                Full_locs = np.vstack((R_full, R_K_subj))
                RBF_weights = rbf(R_full, R_K_subj, r)
                C_K_subj[np.eye(C_K_subj.shape[0]) == 1] = 0
                C_K_subj[np.where(np.isnan(C_K_subj))] = 0
                Z_K_subj = r2z(C_K_subj)
                Ks, Ws = expand_corrmat_parsed(RBF_weights, Z_K_subj)
                S_expand = Ks / Ws

                S_matrix = squareform(S_expand, checks=False)
                S_weights = ~np.isnan(S_matrix)
                S_weights = S_weights.astype(int)
                Ave_matrix = Ave_data['matrix_sum'].flatten()
                Ave_weights = Ave_data['weights_sum']

                Model_matrix = np.nansum(np.dstack((S_matrix, Ave_matrix)),2)
                Model_weights = S_weights + Ave_weights



                Model = squareform(Model_matrix.flatten()/Model_weights, checks= False)
                Model[np.eye(Model.shape[0]) == 1] = 0
                Model[np.where(np.isnan(Model))] = 0
                ## expand the altered average matrix to the new full set of locations (R_full + R_K_subj)
                RBF_weights = rbf(Full_locs, R_full, r) # 3 by number of good channels

                #### to test if the same expanding:

                Ka, Wa = expand_corrmat_parsed(RBF_weights, Model)
                Ave_expand = z2r(Ka / Wa)
                Ave_expand[np.where(np.isnan(Ave_expand))] = 1
                known_inds, unknown_inds = known_unknown(Full_locs, R_K_subj)
                unknown_timeseries = np.squeeze(np.dot(np.dot(Ave_expand[unknown_inds, :][:, known_inds], np.linalg.pinv(Ave_expand[known_inds, :][:, known_inds])), zscore(npz_data['Y'])[:, k_flat].T).T)
                unknown_df = pd.DataFrame(unknown_timeseries.T, index=unknown_inds)
                unknown_df.to_csv(os.path.join(check_dir, 'gif_df.csv'))
                outfile = os.path.join(check_dir, 'gif_df.mat')
                #scipy.io.savemat(outfile, {'Y_recon': unknown_df, 'R': R_full})


            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
