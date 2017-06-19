#!/usr/bin/env python

import numpy as np
import pandas as pd
import glob
import scipy.io
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix, r2z, z2r, expand_corrmat_j, compute_coord
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, known_unknown, slice_list
import sys
import re
from scipy.stats import zscore
from plot import compare_matrices, plot_cov



## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, r, k_thresh, total_chunks):

    lower_time_gif = 0
    upper_time_gif = 10
    ###################################################################################
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    #loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    npz_data = np.load(fname, mmap_mode='r')
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')

    gif_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'gif_'+ file_name +'_' + str(lower_time_gif)+ '_' + str(upper_time_gif))
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)


    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(gif_dir, 'gif'+ file_name +'_' + str(lower_time_gif)+ '_' + str(upper_time_gif) + '.mat')):

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

                def key_func(x):
                    return os.path.split(x)[-1]

                ###
                files = sorted(glob.glob(os.path.join(gif_dir, '*.npy')), key=key_func)
                results = []
                count = 0
                chunk = []
                for i in files:
                    print i
                    matrix_variables = (file_name + '_k' + str(k_thresh) + '_r' + str(r) + '_pooled_matrix_')
                    match = re.search(matrix_variables, i)
                    if match:
                        count += 1
                        if os.path.basename(i).count('_') == 5:
                            num_chunk = int(os.path.splitext(os.path.basename(i))[0].split("_", 5)[5])
                        elif os.path.basename(i).count('_') == 6:
                            num_chunk = int(os.path.splitext(os.path.basename(i))[0].split("_", 6)[6])
                        else:
                            return "error parsing chunk"
                        data = np.load(i, mmap_mode='r')
                        if np.shape(results)[0] == 0:
                            results = data
                            chunk = num_chunk
                        else:
                            results = np.append(results, data)
                            chunk = np.append(chunk, num_chunk)
                    else:
                        pass
                if not int(total_chunks) == 1:
                    expected_chunks = set(range(0, int(total_chunks)))
                    actual_chunks = set(chunk)
                    dif_chunk = actual_chunks.union(expected_chunks) - actual_chunks.intersection(expected_chunks)
                    for d in dif_chunk:
                        print('error: missing chunk file ' + file_name + '_k' + str(k_thresh) + '_r' + str(
                            r) + '_pooled_matrix_' + str(d).rjust(5, '0'))

                #### this expands the list from the mulitprocessor output  - the lower triangle of the matrix
                Ave_expand = expand_matrix(results, Full_locs)
                Ave_expand[np.where(np.isnan(Ave_expand))] = 1

                outfile = os.path.join(gif_dir,
                                       'ave_results_' + file_name + '_k' + str(k_thresh) + '_r' + str(r))
                np.savez(outfile + '.npz', results=results, n=count)
                outfile = os.path.join(gif_dir, 'full_matrix_' + file_name + '_r_' + str(r) + '.png')
                plot_cov(Ave_expand, outfile=outfile)
                ## reconstruct
                known_inds, unknown_inds = known_unknown(Full_locs, R_K_subj)
                unknown_timeseries = np.squeeze(np.dot(
                    np.dot(Ave_expand[unknown_inds, :][:, known_inds], np.linalg.pinv(Ave_expand[known_inds, :][:, known_inds])),
                    zscore(npz_data['Y'][range(lower_time_gif, upper_time_gif), :])[:, k_flat].T).T)
                unknown_df = pd.DataFrame(unknown_timeseries.T, index=unknown_inds)
                unknown_df.to_csv(os.path.join(gif_dir, 'gif_df.csv'))
                outfile = os.path.join(gif_dir, 'gif_df.mat')
                scipy.io.savemat(outfile, {'Y_recon': unknown_df, 'R': R_full})


            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print('gif exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])