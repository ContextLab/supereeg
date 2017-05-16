#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, partition_jobs
import sys
from joblib import Parallel, delayed
import multiprocessing


## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, matrix_chunk, r, k_thresh):
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    # loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs')

    ## check if cor_fig and full directories exist
    cor_fig_dir = os.path.join(fig_dir, 'sub_corrs')
    if not os.path.isdir(cor_fig_dir):
        os.mkdir(cor_fig_dir)

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_Rstd')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0') + '.npy')):
        #### we are using a different set of locations, downsampled from a standard brain - this would be to create it from the union of all electrodes passing kthresh
        # if not os.path.isfile(os.path.join(get_parent_dir(os.getcwd()), k_loc_name)):
        #     k_full_locs = np.empty(shape=[0, 3])
        #     files = glob.glob(os.path.join(corr_dir, '*.npz'))  ## need to replace this - not sure if what should be in here
        #     for i in files:
        #         try:
        #             sub_data = np.load(i, mmap_mode='r')
        #             R_K_subj, C_K_subj = good_chans(sub_data['K_subj'], sub_data['R_subj'], k_thresh,
        #                                             C = sub_data['C_subj'])
        #             if R_K_subj.shape[0] > 1:
        #                 k_full_locs = np.append(k_full_locs, R_K_subj, axis=0)
        #             else:
        #                 pass
        #         except:
        #             pass
        #     unique_full_locs = np.vstack(set(map(tuple, k_full_locs)))
        #     np.save(os.path.join(get_parent_dir(os.getcwd()), k_loc_name), unique_full_locs[unique_full_locs[:, 0].argsort(),])

        ## load subject's electrodes
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj']
        C_subj = sub_data['C_subj']
        K_subj = sub_data['K_subj']
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
        # R_K_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))

        # index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj= good_chans(K_subj, R_subj, k_thresh, C = C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                ## create covariance matrices with kurtosis thresholed data
                # C_expand = expand_corrmat(R_full, R_K_subj, C_K_subj, float(r))
                ######### to use parallelize forloop:
                #### create weights matrix
                RBF_weights = rbf(R_full, R_K_subj, r)
                #### compile all pairs of coordidnates - loop over R_full matrix (lower triangle)
                # inputs = [(x, y) for x in range(R_full.shape[0]) for y in range(x)]
                # num_cores = multiprocessing.cpu_count()
                # #### can't have the delayed(expand_corrmat) in a function
                # results = Parallel(n_jobs=num_cores)(
                #     delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in inputs)
                # outfile = os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(
                #     r) + '_all')
                # np.save(outfile, results)
                ### try distributed by rows:
                # sliced_up = slice_list(range(R_full.shape[0]), 4)
                # #m_slice = np.logspace(3, 0, 30, dtype='int')
                # inputs = [(x, y) for x in range(np.min(sliced_up[int(matrix_chunk)]),np.max(sliced_up[int(matrix_chunk)])+1, 1) for y in range(x)]


                ### slice list of coordinates in a number of sublists (this shouldn't be hardcoded) and index a slice according to matrix_chunk
                #### this step on my computer takes about 20GB and 12 minutes for a full matrix width of 20,000
                sliced_up = slice_list([(x, y) for x in range(R_full.shape[0]) for y in range(x)], 5000)[int(matrix_chunk)]

                ### With andy's help, partition_jobs
                num_cores = multiprocessing.cpu_count()
                #### can't have the delayed(expand_corrmat) in a function
                # results = Parallel(n_jobs=num_cores)(
                #     delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in partition_jobs(R_full.shape[0])[int(matrix_chunk)])
                results = Parallel(n_jobs=num_cores)(
                    delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in sliced_up)
                outfile = os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0'))
                np.save(outfile, results)

            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0'), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])