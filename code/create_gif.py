#!/usr/bin/env python

from stats import tal2mni, good_channels_loc, time_by_file_index_chunked
from bookkeeping import row_in_array, remove_electrode, known_unknown, alter_avemat, get_parent_dir, get_grand_parent_dir
import numpy as np
import os
import sys





## input for this: file name, electrode location, r and k_thresh
#### so ill have to input electrode location from the cruncher file - just loop through R_K_full

def main(fname, electrode_loc, r, k_thresh):
    ################################################################################
    ### change timeseries = True if you want timeseries not just correlation
    timeseries = True
    ################################################################################
    file_name = os.path.splitext(os.path.basename(fname))[0]
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_matrices')
    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices')

    k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    #loc_name = 'R_full_MNI.npy'

    if not os.path.isfile(
            os.path.join(average_dir, 'average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')):
        print('average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz', 'does not exist')
    # average_matrix(full_dir, r, k_thresh)

    compare_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'correlations')
    if not os.path.isdir(compare_dir):
        os.mkdir(compare_dir)

    gif_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'correlations/gif')
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)


    if not (os.path.isfile(os.path.join(gif_dir, 'corr_timeseries_' + file_name + "_" + electrode_loc + "_" + str(k_thresh) + '_' + str(r) + '.npz'))):
        ## load in: data = subject's electrodes; sub_data = subject kurtosis values; R_full = full electrode matrix (loc_name hardcoded)
        data = np.load(fname, mmap_mode='r')
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
        R_K_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))

        ## convert subject electrodes to MNI space
        R_subj = tal2mni(data['R'])

        ## find good electrodes from defined kurtosis threshold
        R_K_subj, k_flat_removed = good_channels(sub_data['K_subj'], R_subj, k_thresh)

        ## check if none or only one electrode passes kurtosis threshold
        if not R_K_subj == []:
            if np.shape(R_K_subj)[0] > 1:
                ## check if the removed electrode survives kurtosis thresholding
                if row_in_array(R_K_subj, electrode):
                    ## remove electrode from subject level kurtosis array
                    R_K_removed = remove_electrode(R_K_subj, R_subj, electrode)
                    ## then index full array
                    ## inds after kurtosis threshold: known_inds = known electrodes; unknown_inds = all the rest; rm_unknown_ind = where the removed electrode is located in unknown subset
                    known_inds, unknown_inds, electrode_ind = known_unknown(R_K_full, R_K_removed, R_subj, electrode)
                    ## load average correlation matrix and index based on kurtosis thresholded indexing
                    Ave_data = np.load(os.path.join(average_dir, 'average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'),mmap_mode='r')
                    Full_data = np.load(os.path.join(full_dir,'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npz'), mmap_mode='r')
                    Ave_rm_subj = alter_avemat(Ave_data, Full_data)
                    coord = R_subj[int(electrode)]
                    ##################### circular way (subject data not removed) ###################################################
                    #corrs= time_by_file_index(fname, Ave_data, known_inds, unknown_inds, electrode_ind, k_flat_removed, electrode)
                    ##################### non-circular way (removing subject's covariance matrix from average matrix ################
                    outfile = os.path.join(gif_dir,
                                           'corr_timeseries_T_' + file_name + "_" + electrode + "_" + str(
                                               k_thresh) + '_' + str(r) + '.npz')
                    corrs = time_by_file_index_chunked(fname, Ave_rm_subj, known_inds, unknown_inds, electrode_ind, k_flat_removed, electrode, time_series=True)
                    np.savez(outfile, coord=coord, corrs=corrs)

                else:
                    print("electrode_" + electrode + " does not pass k = " + str(k_thresh))
            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print('corr_timeseries_' + file_name + "_" + electrode + "_" + str(k_thresh) + '_' + str(r) + '.npz', 'exists')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])