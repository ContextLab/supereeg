import numpy as np
import numpy.matlib as mat
from scipy.stats import kurtosis, zscore
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
from scipy import linalg
from sklearn.decomposition import PCA

#
# def prettify():
#     sns.despine()
#     plt.tight_layout()

######## from full cov mats ########################
##### try linear algebra way ##############
# tupper = np.triu(r2z(C_K_subj))
# tupper[np.isnan(tupper)] = 0
# C_w = np.dot(RBF_weights, tupper)
# weighted_corr_matrix = np.dot(C_w, C_w.T)
# WTW = np.dot(RBF_weights, RBF_weights.T)
# WTW_trunc = truncate(WTW)
# temp = weighted_corr_matrix/WTW_trunc
###################################################

# fig, ax = plt.subplots()
# mpl.rcParams['axes.facecolor'] = 'white'
# df_corrs = all_corrs.groupby('Subject')['Correlation']
# bin_values = np.arange(start=-1, stop=1, step=.025)
# ax = df_corrs.plot(kind='hist', bins=bin_values, alpha=.5, lw=3, title = 'Correlations with RBF radius = ' + str(r))
# vals = ax.get_yticks()
# ax.set_yticklabels([round_it(x / n_count, 3) for x in vals])
# ax.set_ylabel('Proportion of electrodes, grouped by subject')
# ax.set_xlabel('Correlation')
# ax.set_xlim(-1, 1)
# outfile = os.path.join(fig_dir, 'hist_color_' + str(k_thresh) + '_' + str(r) + '.pdf')
# plt.savefig(outfile)
# plt.clf()

######### Plot Density BNV ####################################
# node_matrix = brain_net_viewer_format_color(np.asarray(all_corrs['R'].tolist()), np.asarray(all_corrs['density']))
# np.savetxt(os.path.join(fig_dir, 'electrode_density.node'), node_matrix)

#
# ######## Create accuracy vs samplerate ####################
# outfile = os.path.join(fig_dir, 'samplerate_acc_scatter_' + file_name + '_electrode_' + str(electrode) + '_' +  str(k_thresh) + '_' + str(r) + '.pdf')
# df_sr = all_corrs.groupby(['Subject']).mean().reset_index()
# df_sr['chosen_one'] = df_sr.Subject == file_name
# scatter_plot(df_sr, 'Sample rate', 'Correlation', plot_type = 'scatter', outfile = outfile,  color = 'k', alpha = .6, subject = file_name)
# plt.clf()
# outfile = os.path.join(fig_dir, 'samplerate_acc_swarm_' + file_name + '_electrode_' + str(electrode) + '_' + str(k_thresh) + '_' + str(r) + '.pdf')
# ax = sb.swarmplot(df_sr['Sample rate'], df_sr['Correlation'], hue=df_sr['chosen_one'], palette = {'k', 'r'})
# ax.legend_.remove()
# plt.savefig(outfile)

# ######## Create vector for brain data - with new val_interp function instead of RBF ################
#
# # R_corrs = val_interp(R_std, 'Correlation', all_corrs, tau = .5, fishersZ=True)
# # flattened_corr = flatten_arrays(R_corrs.Correlation.values.flatten())
# # scipy.io.savemat(os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_pltr_' + str(plot_r) + '.mat'), dict(R=R_std, V=np.atleast_2d(flattened_corr)))
# # R_density = val_interp(R_std, 'Correlation', all_corrs, tau=.5, fishersZ=False)
# # flattened_density = flatten_arrays(R_density.Correlation.values.flatten())
# # scipy.io.savemat(os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_pltr_' + str(plot_r) + '.mat'), dict(R=R_std, V=np.atleast_2d(flattened_density)))
#
# # ######## try parallel processing ##############################################
# # inputs = [(a_row) for a_row in R_std[0:np.shape(R_std)[0]:1, :]]
# # num_cores = multiprocessing.cpu_count()
# # print(num_cores)
# # # ## can't have the delayed(processInput) in a function
# # R_corrs = Parallel(n_jobs=num_cores)(
# #     delayed(val_interp_parallel)(a_row, 'Correlation', all_corrs, tau = 1, fishersZ=True) for a_row in inputs)
# # flattened_corr = flatten_arrays(R_corrs)
# # scipy.io.savemat(
# #     os.path.join(fig_dir, 'mean_corr_k_' + str(k_thresh) + '_r_' + str(r) + '_tau_' + str(1) + '.mat'),
# #     dict(R=R_std, V=np.atleast_2d(flattened_corr)))
# # R_density = Parallel(n_jobs=num_cores)(
# #     delayed(val_interp_parallel)(a_row, 'Density', all_corrs, tau = 1, fishersZ=False) for a_row in inputs)
# # flattened_density = flatten_arrays(R_density)
# # scipy.io.savemat(
# #     os.path.join(fig_dir, 'mean_dense_k_' + str(k_thresh) + '_r_' + str(r) + '_tau_' + str(1) + '.mat'),
# #     dict(R=R_std, V=np.atleast_2d(flattened_density)))
# ############### RBF plot ################################################
# if not os.path.isfile(os.path.join(fig_dir, 'RBF_r' + str(r) + '.png')):
#     outfile = os.path.join(fig_dir, 'RBF.png')
#     center = np.atleast_2d([0, 0])
#     coords = []
#     fun_map = np.empty((1000,1000))
#     for x in np.linspace(-2,2,1000):
#         for y in np.linspace(-2,2,1000):
#             coordinate = (x, y)
#             coords.append(coordinate)
#
#     weight = rbf(coords, center, 1)
#     weights = weight.reshape(1000, 1000)
#     sb.heatmap(weights, cmap="RdBu_r")
#     plt.savefig(outfile)
#     plt.clf()

# b = BrainData(os.path.join(get_parent_dir(os.getcwd()), 'avg152T1_gray_3mm.nii.gz'))

###### find number of samples - this is now found when creating all_corrs.csv
# DF_samples = pd.DataFrame()
# files = glob.glob(os.path.join(corr_dir, '*.npz'))
# for i in files:
#
#     def parse_path_name(path_name):
#         if os.path.basename(path_name).count('_') == 2:
#             f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 2)[2]
#             return f_name
#         elif os.path.basename(path_name).count('_') == 3:
#             f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 3)[2:4])
#             return f_name
#         else:
#             return "error"
#
#     f_name = parse_path_name(i)
#     sub_data = np.load(i, mmap_mode='r')
#     npz_data = np.load(os.path.join(path_to_npz, f_name + '.npz'), mmap_mode='r')
#     K_subj = sub_data['K_subj']
#     k_flat = np.squeeze(np.where(K_subj < int(k_thresh)))
#     temp_pd = pd.DataFrame()
#     if not k_flat == []:
#         if np.shape(k_flat)[0] > 1:
#             npz_data = np.load(os.path.join(path_to_npz, file_name + '.npz'), mmap_mode='r')
#             tempsamples = np.shape(npz_data['Y'])[0]
#             tempsamplerate = np.mean(npz_data['samplerate'])
#             temp_pd = pd.DataFrame({'Subject': [f_name],'Sample rate': [tempsamplerate], 'Samples': [tempsamples]})
#         else:
#             print('noo')
#     if DF_samples.empty:
#         DF_samples = temp_pd
#     else:
#         DF_samples = DF_samples.append(temp_pd)
# data = DF_samples
# a_dict = {col_name: data[col_name].values for col_name in data.columns.values}
# outfile = os.path.join(fig_dir, 'samples.mat')
# scipy.io.savemat(outfile, {'struct': a_dict})

#### not sure i'll ever need this code:
# for R, corr in zip(tempR, tempcorr):
#     for i, row in compile_DF.iterrows():
#         if np.array_equal(row['R'], R):
#             nn_corr.append(corr)
#             se_corr.append(compile_DF.reset_index().ix[i]['Correlation'])
#         else:
#             pass
#     #compile_DF.ix[compile_DF['R'] = R, 'nn_corr'] == corr
#     nn_corr = []
#     # for ix, row in compile_DF.iterrows():
#     #     [row['nn_corr'][row['R'] == f] = b
#     #     row.ix[row['A'] = df['B'], 'C'] == 0

###############################################################
############# from stats.py ##################################
###############################################################
def pca_describe_chunked(fname, k_inds, field='Y'):
    """
    Function that calculates average PCA

    Parameters
    ----------
    fname :  Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    k_inds : list
        Kurtosis passed indices

    Returns
    ----------
    results: pandas dataframe
        The average PCA by session


    """

    def PCA_summary(x):
        cov_alldims = pdist(x, 'correlation')
        list_pca = []
        list_c = []
        # to test: for i in range(2, 6): k_inds.shape[0]
        for c in range(2, x.shape[1]):
            list_pca.append(np.corrcoef(cov_alldims, pdist(PCA(n_components=c).fit_transform(x)))[0][1])
            list_c.append(c)
        return list_pca, list_c

    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])

    results = pd.DataFrame()
    for i in file_inds:
        if np.shape(data['fname_labels'])[1] == 1:
            fname_labels = data['fname_labels'].T
        else:
            fname_labels = data['fname_labels']
        next_inds = np.where(fname_labels == i)[1]
        ### this code should incorporate the average voltage of the known (subject) electrodes and the average for the unknown (the other subjects)
        block_results = pd.DataFrame()
        ### right now, this doesn't use an overlap in time, but this needs to be addressed when I see edge effects
        for each in chunker(next_inds, 10000):
            ### this code should incorporate the average voltage of the known (subject) electrodes and the average for the unknown (the other subjects)
            next_pca, next_c = PCA_summary(np.float32(data[field][:, k_inds][filter(None, each), :]))
            next_compare_time = pd.DataFrame({'comp_num': next_c, 'PCA': next_pca, 'session': i})
            if block_results.empty:
                block_results = next_compare_time
            else:
                block_results = block_results.append(next_compare_time)
        if results.empty:
            results = block_results
        else:
            results = results.append(block_results)
    return results.groupby(['session', 'comp_num'])[['PCA']].mean().reset_index()


################### NOT USED #####################
def time_by_file_index(fname, ave_data, known_inds, unknown_inds, electrode_ind, k_flat_removed, electrode, field='Y',
                       ave_field='average_matrix'):
    """
    Session dependent function that calculates that finds the correlation of the predicted and actual timeseries for a given location

    Parameters
    ----------
    fname : Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    ave_data: array
        Average correlation matrix

    known_inds: list
        Indices for known electrodes in average matrix

    unknown_inds: list
        Indices for unknown electrodes in average matrix

    electrode_ind: int
        Index for estimated location in average matrix

    k_flat_removed: list
        Indices of good channels (pass kurtosis test) in Y

    Returns
    ----------
    results : numpy array
         Correlation between predicted and actual timeseries by session

    """
    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])

    results = []
    for i in file_inds:
        if np.shape(data['fname_labels'])[1] == 1:
            fname_labels = data['fname_labels'].T
        else:
            fname_labels = data['fname_labels']
        next_inds = np.where(fname_labels == i)[1]
        ### this code should incorporate the average voltage of the known (subject) electrodes and the average for the unknown (the other subjects)
        next_predicted = np.squeeze(np.dot(np.dot(np.float32(ave_data[ave_field][unknown_inds, :][:, known_inds]),
                                                  np.linalg.pinv(
                                                      np.float32(ave_data[ave_field][known_inds, :][:, known_inds]))),
                                           zscore(np.float32(data[field][next_inds, :])[:, k_flat_removed]).T).T[:,electrode_ind])
        next_actual = np.squeeze(zscore(np.float32(data[field][:, [int(electrode)]])[next_inds, :]))
        next_compare_time = pd.DataFrame({'actual': next_actual, 'predicted': next_predicted, 'session': i})
        next_R = next_compare_time.corr()['actual']['predicted']
        if results == []:
            results = next_R
        else:
            results = np.vstack((results, next_R))
    return results


################### NOT USED #####################
def pca_description(fname):
    """
    Function that calculates average PCA - NOT ACTUALLY USED SINCE THERES AN ISSUE WITH AGGREGATE - instead use pca_describe_var

    Parameters
    ----------
    fname :  Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    Returns
    ----------
    results: array
        The average PCA by session


    """
    def aggregate(prev, next):
        return np.sum(np.vstack((prev, next)), axis=0)
    ### issue with aggregate - different numbers of PCs based on different sessions

    def PCA_summary(x):
        cov_alldims = pdist(x,'correlation')
        list_pca = []
        for i in range(2, x.shape[1]):
            list_pca.append(np.corrcoef(cov_alldims, pdist(PCA(n_components=i).fit_transform(x)))[0][1])
        return list_pca

    pca_attr = apply_by_file_index(fname, PCA_summary, aggregate)
    n = n_files(fname)

    return pca_attr/n


def truncate(m, tol=1e-10):
    """
    Truncate float

    Parameters
    ----------
    m : float
        Number to be truncated


    Returns
    ----------
    m : float
        Truncated number

    """
    m[np.abs(m) <= tol] = 0
    return m

### this was replaced by processInput for multiprocessing ######
def expand_corrmat(R_full, R_sub, C_sub, r):
    """
    Finds unique rows
    Parameters
    ----------
    R_full : array
        locations to at which to reconstruct correlation matrix

    R_sub : array
        locations where we have samples

    Returns
    ----------
    results : array
        unique rows

    """
    # R_full: locations to reconstruct correlation matrix at
    # R_sub: locations where we have samples
    # C_sub: correlation matrix where we have samples
    # r: width of RBF used to weight nearby locations

    # loop through each pair of location (x, y) in R_full.  The correlation at (x,y) in C_full is a weighted sum of the
    # correlations at each pair of locations (i, j) in R_sub, where the weights are given by RBF(x, i, r)*RBF(y, j, r).
    RBF_weights = rbf(R_full, R_sub, r)

    C_full = np.zeros([R_full.shape[0], R_full.shape[0]])
    for x in range(R_full.shape[0]):
        for y in range(x):
            weighted_sum = 0
            sum_of_weights = 0

            for i in range(R_sub.shape[0]):
                for j in range(i):
                    next_weight = RBF_weights[x, i] * RBF_weights[y, j]
                    weighted_sum += r2z(C_sub[i, j]) * next_weight
                    sum_of_weights += next_weight

            C_full[x, y] = z2r(weighted_sum / sum_of_weights)

    return C_full + C_full.T + np.eye(C_full.shape[0])

def brain_vector(Full_df, feature, std_locs, radius):
    tempR = np.atleast_2d(Full_df['R'])
    tempmeanfeature = Full_df[feature]
    return r2z(tempmeanfeature) * rbf(std_locs, tempR, radius).T

### the following three functions are apart of the interpolating values in the standard brain - use matlab code now, since this takes forever to run
def sim(a,x,tau):
    return np.exp(-tau * cdist(a,x))


def val_interp(all_locs, feature, Full_df, tau = None, fishersZ = False):
    if tau is None:
        tau = 1
    R_df = pd.DataFrame()
    for a_row in all_locs:
        a_row = a_row
        weighted_corr_sum = 0
        sims_sum = 0
        for index, x_row in Full_df.iterrows():
            e_row = x_row['R']
            tempsim = sim(np.atleast_2d(a_row), np.atleast_2d(e_row), tau).flatten()
            if fishersZ is True:
                weighted_corr_sum += tempsim * z2r(x_row[feature])
            else:
                weighted_corr_sum += tempsim * x_row[feature]
            sims_sum += tempsim
        if fishersZ is True:
            R_df_temp = pd.DataFrame({'R': [a_row], feature: [z2r(weighted_corr_sum / sims_sum)]}, index=[0])
        else:
            R_df_temp = pd.DataFrame({'R': [a_row], feature: [weighted_corr_sum / sims_sum]}, index=[0])
        if R_df.empty:
            R_df = R_df_temp
        else:
            R_df = R_df.append(R_df_temp)
    return R_df

def val_interp_parallel(a_row, feature, Full_df, tau = None, fishersZ = False):
    if tau is None:
        tau = 1
    weighted_corr_sum = 0
    sims_sum = 0
    for index, x_row in Full_df.iterrows():
        e_row = x_row['R']
        tempsim = sim(np.atleast_2d(a_row), np.atleast_2d(e_row), tau).flatten()
        if fishersZ is True:
            weighted_corr_sum += tempsim * z2r(x_row[feature])
        else:
            weighted_corr_sum += tempsim * x_row[feature]
        sims_sum += tempsim
    if fishersZ is True:
        return z2r(weighted_corr_sum / sims_sum)
    else:
       return weighted_corr_sum / sims_sum


def extract_pca(Corr_timeseries):
    """
        Compiles PCA values - as well as other subject/electrode specific paramters - creates the compiled pandas dataframe used for figures

        Parameters
        ----------

        Corr_timeseries : npz file
            npz file containing correlation values (loop outside - for each electrode)

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode, correlation, samples, and sample rate

        """
    corr_data = np.load(Corr_timeseries, mmap_mode='r')
    PCA = corr_data['PCA']
    return pd.DataFrame({'sessions': PCA[:, 0], 'Components': PCA[:, 1], 'Var': PCA[:, 2]})

### this has been consolidated into good_chans
def good_channels(k, R, C, k_thresh):
    """
    Finds channels that pass kurtosis test

    Parameters
    ----------
    k : array
        Kurtosis values for each channel for subject -  R_K_subj

    R : array
        Subject's coordinates - R_subj

    C : array
        Subject's correlation matrix

    k_thresh : int
        Kurtosis threshold

    Returns
    ----------
    C : array
        Subject's correlation matrix with kurtosis failed channels removed

    R : array
        Subject's coordinates with kurtosis failed channels removed


    """
    k_flat = np.squeeze(np.where(k < int(k_thresh)))
    R = R[k_flat, :]
    C = C[k_flat, :][:, k_flat]
    return C, R

def good_channels_loc(k, R, k_thresh, electrode):
    """
    Finds channels that pass kurtosis test - and removes electrode location from index

    Parameters
    ----------
    k : array
        Kurtosis values for each channel for subject -  R_K_subj

    R : array
        Subject's coordinates - R_subj

    k_thresh : int
        Kurtosis threshold

    electrode : int
        Index for electrode to be removed

    Returns
    ----------

    R : array
        Subject's coordinates with kurtosis failed channels removed

    k_flat_removed : list
        Indices that pass kurtosis test with electrode location removed

    """
    k_flat = np.squeeze(np.where(k < int(k_thresh)))
    k_flat_removed = np.delete(k_flat, np.where(k_flat == int(electrode)), 0)
    R = R[k_flat, :]
    return R, k_flat_removed


def flatten_arrays(array_of_arrays):
    """


        Parameters
        ----------
        array_of_arrays : ndarray
            arrays to flatten

        Returns
        ----------
        results : list
            Flattened array

        """
    flattened_array = []
    for sublist in array_of_arrays:
        for item in sublist:
            flattened_array.append(item)
    return flattened_array


############# For Loop Way ######################################
#### compile all pairs of coordidnates - loop over R_full matrix (lower triangle)
inputs = [(x, y) for x in range(R_full.shape[0]) for y in range(x)]
num_cores = multiprocessing.cpu_count()
#### can't have the delayed(expand_corrmat) in a function
results = Parallel(n_jobs=num_cores)(
    delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in inputs)
outfile = os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(
    r) + '_for_loop')
np.save(outfile, results)

temp = expand_matrix(results, R_full)
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
sb.heatmap(temp, ax=ax1)

############## Sliced For Loop Way ####################################
### slice list of coordinates in a number of sublists (this shouldn't be hardcoded) and index a slice according to matrix_chunk
#### this step on my computer takes about 20GB and 12 minutes for a full matrix width of 20,000
sliced_up = slice_list([(x, y) for x in range(R_full.shape[0]) for y in range(x)], 4)[int(matrix_chunk)]
### With andy's help, partition_jobs
num_cores = multiprocessing.cpu_count()
#### can't have the delayed(expand_corrmat) in a function
# results = Parallel(n_jobs=num_cores)(
#     delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in partition_jobs(R_full.shape[0])[int(matrix_chunk)])
results = Parallel(n_jobs=num_cores)(
    delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in sliced_up)
outfile = os.path.join(full_dir,
                       file_name + '_k' + str(k_thresh) + '_r' + str(r) + '_pooled_matrix_' + matrix_chunk.rjust(5,
                                                                                                                 '0'))
np.save(outfile, results)
############# Linear Algebra Way ####################################
## attempt 1:
z_c = r2z(C_K_subj)
z_c[np.isnan(z_c)] = 0
C_w = np.dot(RBF_weights, z_c)
weighted_corr_matrix = np.dot(C_w, RBF_weights.T)
WTW = np.dot(RBF_weights, RBF_weights.T)
temp1 = z2r(weighted_corr_matrix / WTW)
sb.heatmap(temp1, ax=ax2)
WTW = np.sum(np.dot(RBF_weights, RBF_weights.T))
temp2 = z2r(weighted_corr_matrix / WTW)
sb.heatmap(temp2, ax=ax3)

## attempt 2:
z_c = r2z(C_K_subj)
z_c[np.isnan(z_c)] = 0
C_w = np.dot(RBF_weights, z_c)
weighted_corr_matrix = np.dot(C_w, C_w.T)
WTW = np.dot(RBF_weights, RBF_weights.T)
temp2 = z2r(weighted_corr_matrix / WTW)
sb.heatmap(temp2, ax=ax3)

## attempt 3:
tupper = np.triu(r2z(C_K_subj))
tupper[np.isnan(tupper)] = 0
C_w = np.dot(RBF_weights, tupper)
# weighted_corr_matrix = np.dot(C_w, RBF_weights.T)
weighted_corr_matrix = np.dot(C_w, C_w.T)
WTW = np.dot(RBF_weights, RBF_weights.T)
temp3 = z2r(weighted_corr_matrix / WTW)
sb.heatmap(temp3, ax=ax4)

## attempt 4:
tupper = np.triu(r2z(C_K_subj))
tupper[np.isnan(tupper)] = 0
C_w = np.dot(RBF_weights, tupper)
# weighted_corr_matrix = np.dot(C_w, RBF_weights.T)
weighted_corr_matrix = np.dot(C_w, RBF_weights.T)
WTW = np.dot(RBF_weights, RBF_weights.T)
temp4 = z2r(weighted_corr_matrix / WTW)
sb.heatmap(temp4, ax=ax5)

DF = pd.DataFrame({'temp': np.ravel(temp), 'temp1': np.ravel(temp1), 'temp2': np.ravel(temp2),
                   'temp3': np.ravel(temp3)})


# eng = matlab.engine.start_matlab()
# eng.create_supereeg_interpmap_time('gif_df.mat', 1)
# gif_files = glob.glob(os.path.join(gif_dir, 'gif_frame_t_*'))
# for g in gif_files:
#     gif_name = os.path.splitext(os.path.basename(g))[0]
#     outfile = os.path.join(gif_png_dir, str(gif_name)+'.png')
#     ni_plt.plot_glass_brain(g, display_mode='lyrz', cmap='gray',threshold = 0, plot_abs=False, colorbar='True',
#                          output_file=outfile)