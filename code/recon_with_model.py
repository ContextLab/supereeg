#!/usr/bin/env python

import numpy as np
import pandas as pd
import glob
import scipy.io
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix, r2z, z2r, expand_corrmat_j, compute_coord
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, known_unknown, slice_list
import sys
from scipy.stats import zscore
from scipy.spatial.distance import squareform as squareform
from joblib import Parallel, delayed
import multiprocessing




## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, matrix_chunk, r, k_thresh, total_chunks):

    lower_time_gif = 0
    upper_time_gif = 10
    ###################################################################################
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    #loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    loc_name = 'R_small_MNI.npy'

    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')

    gif_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'gif_'+ file_name +'_' + str(lower_time_gif)+ '_' + str(upper_time_gif))
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)


    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(gif_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r) + '_pooled_matrix_' + matrix_chunk.rjust(5, '0') + '.npy')):

        ## load subject's correlation matrix and electrodes
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj'] # electrode locations
        C_subj = sub_data['C_subj'] # subject data
        K_subj = sub_data['K_subj'] # kurtosis - 1 by n_elecs
        ## index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj, k_flat = good_chans(K_subj, R_subj, k_thresh, C=C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                ## load full set of electrode locations
                R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
                ## add the full set of locations with the subject's locations
                Full_locs = np.vstack((R_full, R_K_subj))

                ## load average matrix
                Ave_data = np.load(
                    os.path.join(average_dir, 'average_model' + '_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'),
                    mmap_mode='r')
                ## remove subject's full correlation from the average matrix
                Ave_mat = squareform(Ave_data['matrix_sum'].flatten()/Ave_data['weights_sum'], checks= False)

                ## expand the altered average matrix to the new full set of locations (R_full + R_K_subj)
                RBF_weights = rbf(Full_locs, R_full, r) # 3 by number of good channels
                sliced_up = slice_list([(x, y) for x in range(RBF_weights.shape[0]) for y in range(x)], int(total_chunks))[int(matrix_chunk)]

                Ave_mat[np.eye(Ave_mat.shape[0]) == 1] = 0
                Ave_mat[np.where(np.isnan(Ave_mat))] = 0
                results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(compute_coord)(coord, RBF_weights, Ave_mat) for coord in sliced_up)
                outfile = os.path.join(gif_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0'))
                np.save(outfile, results)

            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r) + '_pooled_matrix_' + matrix_chunk.rjust(5, '0'), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])