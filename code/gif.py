
import os
import sys
import scipy.io
from scipy.stats import zscore
import numpy as np
from stats import tal2mni, good_chans
from bookkeeping import get_parent_dir, get_grand_parent_dir, known_unknown
import pandas as pd
import glob as glob
from scipy.spatial.distance import cdist

### input is now: path_to_npz_subject_data electrode_number radius_for_RBF kurtosis_threshold
def main(fname, count, r, k_thresh):

    lower_time_gif = 0
    upper_time_gif = 10
    ###################################################################################
    ################### compile paths to directories: #################################
    path_to_npz =os.path.splitext(os.path.dirname(fname))[0]
    file_name = os.path.splitext(os.path.basename(fname))[0]
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs/paper')
    compare_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'correlations')
    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')
    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_model')
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    gif_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'gif_'+ file_name +'_' + str(lower_time_gif)+ '_' + str(upper_time_gif))
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)

    ## downsampled locations with 5mm resolution:
    #loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    loc_name = 'R_small_MNI.npy'
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'

    npz_data = np.load(fname, mmap_mode='r')
    R_subj = tal2mni(npz_data['R'])
    R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
    sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
    ave_data = np.load(os.path.join(average_dir, 'average_model' + "_k_" + str(k_thresh) + '_r_' + str(r) + '_count_' + str(count) + '.npz'), mmap_mode='r')

    R_K_subj, k_flat = good_chans(sub_data['K_subj'], R_subj, k_thresh)
    known_inds = np.argmin(cdist(R_full, R_K_subj, metric='sqeuclidean'), axis=0)
    unknown_inds = list(set(range(np.shape(R_full)[0])) - set(known_inds))
    #known_inds, unknown_inds = known_unknown(R_full, R_subj)
    known_timeseries = zscore(npz_data['Y'][range(lower_time_gif, upper_time_gif), :][:, k_flat])
    subset_known = ave_data['average_matrix'][known_inds, :][:, known_inds]
    print(subset_known)
    subset_known[np.where(np.isnan(subset_known))] = 0
    unknown_timeseries = np.squeeze(np.dot(np.dot(ave_data['average_matrix'][unknown_inds, :][:, known_inds], np.linalg.pinv(subset_known)), zscore(npz_data['Y'][range(lower_time_gif, upper_time_gif), :])[:, k_flat].T).T)
    unknown_df = pd.DataFrame(unknown_timeseries.T, index=unknown_inds)
    known_df = pd.DataFrame(known_timeseries.T, index=known_inds)
    gif_df = pd.concat([unknown_df, known_df]).sort_index()
    ### this is only because of my nearest neighbors method - this wont be necessary for the expanded corr calc added
    gif_df = gif_df.groupby(gif_df.index).mean()
    no_col = gif_df.as_matrix()
    #gif_df.rename(columns=lambda x: 't_' + str(x).rjust(6, '0'), inplace=True)
    # R = []
    # for row in R_full:
    #     R = np.append(R, str(row))
    # gif_df['R'] = R
    gif_df.to_csv(os.path.join(gif_dir, 'gif_df.csv'))
    outfile = os.path.join(gif_dir, 'gif_df.mat')
    a_dict = {col_name: gif_df[col_name].values for col_name in gif_df.columns.values}
    scipy.io.savemat(outfile, {'Y_recon': no_col, 'R': R_full})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])