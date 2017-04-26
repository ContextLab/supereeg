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



def apply_by_file_index(fname, xform, aggregator, field='Y'):

    """
    Session dependent function application and aggregation

    Parameters
    ----------
    fname : Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    xform : function
        The function to apply to the data matrix from each filename

    aggregator: function
        Tunction for aggregating results across multiple iterations

    Returns
    ----------
    results : numpy array
         Array of aggregated results

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
        next_vals = xform(data[field][next_inds, :])
        if len(results) == 0:
            results = next_vals
        else:
            results = aggregator(results, next_vals)
    return results


def n_files(fname):

    """
    Session count

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
    n : int
        Number of sessions for filename


    """
    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])
    n = len(file_inds)
    return n


def kurt_vals(fname):
    """
    Function that calculates maximum kurtosis values for each channel

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
    results: array (1 x len(R))
        Maximum kurtosis across sessions for each channel


    """
    def aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    return apply_by_file_index(fname, kurtosis, aggregate, field='Y')


def corrmat(fname):
    """
    Function that calculates the average subject level correlation matrix for filename across session

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
    results: squareform matrix
        The average correlation matrix across sessions


    """
    def aggregate(prev, next):
        return np.sum(np.concatenate((prev[:, :, np.newaxis], next[:, :, np.newaxis]), axis=2), axis=2)

    def zcorr(x):
        return r2z(1 - squareform(pdist(x.T, 'correlation')))

    summed_zcorrs = apply_by_file_index(fname, zcorr, aggregate)
    n = n_files(fname)

    return z2r(summed_zcorrs / n)

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

def time_by_file_index_chunked(fname, ave_data, known_inds, unknown_inds, electrode_ind, k_flat_removed, electrode, time_series, field='Y'):
    """
    Session dependent function that calculates that finds either the timeseries or the correlation of the predicted and actual timeseries for a given location chunked by 25000 timepoints

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

    time_series: boolean
        True: output is predicted and actual timeseries
        False: output is predicted and actual correlation

    Returns
    ----------
    results : pandas dataframe
        If timeseries input is:
        True: output is predicted and actual timeseries
        False: output is predicted and actual correlation


    """
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
        for each in chunker(next_inds, 25000):
            ### this code should incorporate the average voltage of the known (subject) electrodes and the average for the unknown (the other subjects)
            next_predicted = np.squeeze(np.dot(np.dot(np.float32(ave_data[unknown_inds, :][:, known_inds]),
                                                      np.linalg.pinv(
                                                          np.float32(
                                                              ave_data[known_inds, :][:, known_inds]))),
                                               zscore(np.float32(data[field][filter(None, each), :])[:,
                                                      k_flat_removed]).T).T[:,
                                        electrode_ind])
            next_actual = np.squeeze(zscore(np.float32(data[field][:, [int(electrode)]])[filter(None, each), :]))
            next_compare_time = pd.DataFrame({'timepoint': filter(None, each), 'actual': next_actual, 'predicted': next_predicted, 'session': i})
            if block_results.empty:
                block_results = next_compare_time
            else:
                block_results = block_results.append(next_compare_time)
        if results.empty:
            results = block_results
        else:
            results = results.append(block_results)
    if not time_series == True:
        return results.groupby('session')[['actual','predicted']].corr().ix[0::2,'predicted'].as_matrix().reshape(np.shape(file_inds)[0], 1)
    return results

################### NOT USED #####################
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


def pca_describe_var(fname, k_inds, field='Y'):
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
    ### output is either timeseries or correlation - change time_series to True for timeseries; False for correlation

    def PCA_summary(x):
        # to test: for i in range(2, 6): k_inds.shape[0]: x.shape[1]
        temp_pca = PCA(n_components=x.shape[1])
        temp_pca.fit(x)
        list_pca = temp_pca.explained_variance_ratio_
        return list_pca

    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])

    results = pd.DataFrame()
    for i in file_inds:
        if np.shape(data['fname_labels'])[1] == 1:
            fname_labels = data['fname_labels'].T
        else:
            fname_labels = data['fname_labels']
        next_inds = np.where(fname_labels == i)[1]
        next_pca = PCA_summary(np.float32(data[field][next_inds, :][:, k_inds]))
        next_compare_time = pd.DataFrame({'PCA': next_pca, 'session': i})
        next_compare_time['comp_num'] = next_compare_time.index + 2
        if results.empty:
            results = next_compare_time
        else:
            results = results.append(next_compare_time)
    return results.groupby(['session', 'comp_num'])[['PCA']].mean().reset_index()



def nearest_neighbors_corr(fname, k_inds, Yfield='Y', Rfield='R'):
    """
    Function that finds the correlation between each electrode and its nearest neighbor

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
        The correlation between timeseries of electrode and nearest neighbor


    """

    def find_nearest(n_by_3_Locs, nearest_n):
        nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
        distances, indices = nbrs.kneighbors(n_by_3_Locs)
        return indices

    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])
    # electrode locations
    R_matrix = tal2mni(data[Rfield])
    # find nearest neighbor index
    nearest_R_inds = find_nearest(R_matrix, 2)

    results = pd.DataFrame()
    for i in file_inds:
        if np.shape(data['fname_labels'])[1] == 1:
            fname_labels = data['fname_labels'].T
        else:
            fname_labels = data['fname_labels']
        next_inds = np.where(fname_labels == i)[1]
        # loop over nearest neighbor index
        timeseries = pd.DataFrame()
        session_results = pd.DataFrame()
        for [a, b] in nearest_R_inds[k_inds]:
            timeseries['this_loc'] = data[Yfield][next_inds, :][:, a]
            timeseries['nearest_loc'] = data[Yfield][next_inds, :][:, b]
            next_R = r2z(timeseries.corr()['this_loc']['nearest_loc'])
            by_session = pd.DataFrame({'Loc_ind':[a], 'Nearest_ind': [b], 'Corr': [next_R], 'session': [i]})
            if session_results.empty:
                session_results = by_session
            else:
                session_results = session_results.append(by_session)
        if results.empty:
            results = session_results
        else:
            results = results.append(session_results)
    return R_matrix[k_inds], results.groupby(['session', 'Loc_ind'])[['Corr']].mean().reset_index().groupby(['Loc_ind'])[['Corr']].mean()


def z2r(z):
    """
    Function that calculates the inverse Fisher z-transformation

    Parameters
    ----------
    z : int or array
        Fishers z transformed correlation value

    Returns
    ----------
    result : int or array
        Correlation value


    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def r2z(r):
    """
    Function that calculates the Fisher z-transformation

    Parameters
    ----------
    r : int or array
        Correlation value

    Returns
    ----------
    result : int or array
        Fishers z transformed correlation value


    """
    return 0.5 * (np.log(1 + r) - np.log(1 - r))


def rbf(x, center, width):
    """
    Radial basis function

    Parameters
    ----------
    x : array
        Series of all coordinates (one per row) - R_full

    c : array
        Series of subject's coordinates (one per row) - R_subj

    width : int
        Radius

    Returns
    ----------
    results : array
        Matrix of RBF weights for each location in R_full


    """
    return np.exp(-cdist(x, center, metric='euclidean') ** 2 / float(width))

def round_it(locs, places):
    """
    Rounding function

    Parameters
    ----------
    locs : float
        Number be rounded

    places : int
        Number of places to round

    Returns
    ----------
    result : float
        Rounded number


    """
    return np.round(locs, decimals=places)



### good channels  needs to be cleaned up - could all be one function !!!


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

def good_chans(k, R, k_thresh, C=None):
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

    k_flat : list
        Indices of channels that pass kurtosis test


    """
    k_flat = np.squeeze(np.where(k < int(k_thresh)))
    R = R[k_flat, :]
    if not C is None:
        C = C[k_flat, :][:, k_flat]
        return C, R
    else:
        return R, k_flat

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


def tal2mni(r):
    """
    Convert coordinates (electrode locations) from Talairach to MNI space

    Parameters
    ----------
    r : array
        Coordinate locations (Talairach space)


    Returns
    ----------
    results : array
        Coordinate locations (MNI space)

    """
    rotmat = np.array([[1, 0, 0, 0], [0, 0.9988, 0.0500, 0], [0, -0.0500, 0.9988, 0], [0, 0, 0, 1.0000]])
    up = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.9200, 0], [0, 0, 0, 1.0000]])
    down = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.8400, 0], [0, 0, 0, 1.0000]])

    inpoints = np.c_[r, np.ones(r.shape[0], dtype=np.float)].T
    tmp = inpoints[2,:] < 0
    inpoints[:,tmp] = linalg.solve(np.dot(rotmat, down), inpoints[:, tmp])
    inpoints[:,~tmp] = linalg.solve(np.dot(rotmat, up), inpoints[:, ~tmp])

    return round_it(inpoints[0:3, :].T,2)

##### i dont think this is used ######
def uniquerows(x):
    """
    Finds unique rows

    Parameters
    ----------
    x : array
        Coordinates


    Returns
    ----------
    results : array
        unique rows

    """
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx = np.unique(y, return_index=True)

    return x[idx]


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


def processInput(coord, R_sub, RBF_weights, C_sub):
    weighted_sum = 0
    sum_of_weights = 0
    for h in range(R_sub.shape[0]):
        for j in range(h):
            next_weight = RBF_weights[coord[0], h] * RBF_weights[coord[1], j]
            weighted_sum += r2z(C_sub[h, j]) * next_weight
            sum_of_weights += next_weight
    return z2r(weighted_sum / sum_of_weights)


def expand_matrix(output_list, R_full):
    output_array = np.array(output_list)
    C_full = np.zeros([R_full.shape[0], R_full.shape[0]])
    C_full[np.tril_indices(R_full.shape[0], -1)] = output_array
    return C_full + C_full.T + np.eye(C_full.shape[0])


def chunker(iterable, chunksize):
        return map(None, *[iter(iterable)] * chunksize)

def brain_vector(Full_df, feature, std_locs, radius):
    tempR = np.atleast_2d(Full_df['R'])
    tempmeanfeature = Full_df[feature]
    return r2z(tempmeanfeature) * rbf(std_locs, tempR, radius).T

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

def compile_corrs(path_to_npz_data, Corr_timeseries):

    def parse_path_name(path_name):
        if os.path.basename(path_name).count('_') == 5:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 5)[2]
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_", 5)[3]
            return f_name, electrode
        elif os.path.basename(path_name).count('_') == 6:
            f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 6)[2:4])
            electrode = os.path.splitext(os.path.basename(path_name))[0].split("_", 5)[4]
            return f_name, electrode
        else:
            return "error"

    f_name, electrode = parse_path_name(Corr_timeseries)
    corr_data = np.load(Corr_timeseries, mmap_mode='r')
    npz_data = np.load(os.path.join(path_to_npz_data, f_name + '.npz'), mmap_mode='r')
    tempR = round_it(corr_data['coord'], 2)
    tempmeancorr = z2r(np.mean(r2z(corr_data['corrs'])))
    tempsamplerate = np.mean(npz_data['samplerate'])
    tempsamples = np.shape(npz_data['Y'])[0]

    return pd.DataFrame({'R': [tempR], 'Correlation': [tempmeancorr], 'Subject': [f_name], 'Electrode': [electrode], 'Sample rate' : [tempsamplerate], 'Samples': [tempsamples]})

def compile_nn_corrs(nn_corr_file):

    def parse_path_name(path_name):
        if os.path.basename(path_name).count('_') == 2:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 2)[0]
            return f_name
        elif os.path.basename(path_name).count('_') == 3:
            f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 3)[0:2])
            return f_name
        else:
            return "error"
    #compile_DF.append()
    f_name = parse_path_name(nn_corr_file)
    nn_corr_data = np.load(nn_corr_file, mmap_mode='r')
    tempcorr = nn_corr_data['nn_corr']
    tempR = nn_corr_data['R_K_subj']
    DF = pd.DataFrame()
    for R, corr in zip(tempR, tempcorr):
        if DF.empty:
            DF = pd.DataFrame({'R': [R], 'Correlation': [corr], 'Subject':[f_name]})
        else:
            DF = DF.append(pd.DataFrame({'R': [R], 'Correlation': [corr], 'Subject':[f_name]}))

    return DF



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



def extract_pca(Corr_timeseries):
    corr_data = np.load(Corr_timeseries, mmap_mode='r')
    PCA = corr_data['PCA']
    return pd.DataFrame({'sessions': PCA[:, 0], 'Components': PCA[:, 1], 'Var': PCA[:, 2]})


def density(n_by_3_Locs, nearest_n):
    nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
    distances, indices = nbrs.kneighbors(n_by_3_Locs)
    return np.exp(-(distances.sum(axis=1) / (np.shape(distances)[1] - 1)))


def normalize_Y(Y_matrix):
    Y = Y_matrix
    m = mat.repmat(np.min(Y, axis = 0), Y.shape[0], 1)
    Y = Y - m
    m = mat.repmat(np.max(Y, axis = 0), Y.shape[0], 1)
    Y = np.divide(Y,m)
    added = mat.repmat(0.5 + np.arange(Y.shape[1]), Y.shape[0], 1)
    Y = Y + added
    return pd.DataFrame(Y)

