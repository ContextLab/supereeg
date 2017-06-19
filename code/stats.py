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
from joblib import Parallel, delayed
import multiprocessing



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
    results : numpy ndarray
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
    results: 1D ndarray(len(R_subj))
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
    results: 2D ndarray(len(R_subj)xlen(R_subj)) matrix
        The average correlation matrix across sessions


    """
    def aggregate(prev, next):
        return np.sum(np.concatenate((prev[:, :, np.newaxis], next[:, :, np.newaxis]), axis=2), axis=2)

    def zcorr(x):
        return r2z(1 - squareform(pdist(x.T, 'correlation')))

    summed_zcorrs = apply_by_file_index(fname, zcorr, aggregate)
    n = n_files(fname)

    return z2r(summed_zcorrs / n)


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

    ave_data: ndarray
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


def find_nearest(n_by_3_Locs, nearest_n):
    nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
    distances, indices = nbrs.kneighbors(n_by_3_Locs)
    return indices

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
    z : int or ndarray
        Fishers z transformed correlation value

    Returns
    ----------
    result : int or ndarray
        Correlation value


    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def r2z(r):
    """
    Function that calculates the Fisher z-transformation

    Parameters
    ----------
    r : int or ndarray
        Correlation value

    Returns
    ----------
    result : int or ndarray
        Fishers z transformed correlation value


    """
    return 0.5 * (np.log(1 + r) - np.log(1 - r))


def rbf(x, center, width):
    """
    Radial basis function

    Parameters
    ----------
    x : ndarray
        Series of all coordinates (one per row) - R_full

    c : ndarray
        Series of subject's coordinates (one per row) - R_subj

    width : int
        Radius

    Returns
    ----------
    results : ndarray(len(R_subj)xlen(R_subj))
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


def good_chans(k, R, k_thresh, *args, **kwargs):
    """
    Finds channels that pass kurtosis test

    Parameters
    ----------
    k : ndarray
        Kurtosis values for each channel for subject -  R_K_subj

    R : ndarray
        Subject's coordinates - R_subj

    C : ndarray
        Subject's correlation matrix

    k_thresh : int
        Kurtosis threshold

    Returns
    ----------
    C : ndarray
        Subject's correlation matrix with kurtosis failed channels removed

    R : ndarray
        Subject's coordinates with kurtosis failed channels removed

    k_flat : list
        Indices of channels that pass kurtosis test


    """
    if 'electrode' in kwargs:
        electrode = kwargs.get('electrode')
        k_flat = np.squeeze(np.where(k < int(k_thresh)))
        k_flat_removed = np.delete(k_flat, np.where(k_flat == int(electrode)), 0)
        R = R[k_flat, :]
        return R, k_flat_removed

    if 'C' in kwargs:
        C = kwargs.get('C')
        k_flat = np.squeeze(np.where(k < int(k_thresh)))
        R = R[k_flat, :]
        C = C[k_flat, :][:, k_flat]
        return R, C, k_flat

    k_flat = np.squeeze(np.where(k < int(k_thresh)))
    R = R[k_flat, :]
    return R, k_flat


def tal2mni(r):
    """
    Convert coordinates (electrode locations) from Talairach to MNI space

    Parameters
    ----------
    r : ndarray
        Coordinate locations (Talairach space)


    Returns
    ----------
    results : ndarray
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

def uniquerows(x):
    """
    Finds unique rows

    Parameters
    ----------
    x : ndarray
        Coordinates


    Returns
    ----------
    results : ndarray
        unique rows

    """
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx = np.unique(y, return_index=True)

    return x[idx]

def expand_corrmat(coord, R_sub, RBF_weights, C_sub):
    """
    This function calculates the RBF weights for each coordinate in the R_full matrix - results are then pooled

    Parameters
    ----------
    coord : ndarray
        Matrix index coordinate pair - looped over each in R_full outside this function

    R_sub : ndarray
        Subject's coordinates - R_subj

    RBF_weights : ndarray
        Weights matrix calculated using rbf function - (len(R_subj)xlen(R_subj)) matrix

    C_sub : ndarray
        Subject level correlation matrix - (len(R_subj)xlen(R_subj)) matrix

    Returns
    ----------
    results : ndarray
        RBF-weighted average at coord (index in full matrix) - results are pooled

    """
    weighted_sum = 0
    sum_of_weights = 0
    for h in range(R_sub.shape[0]):
        for j in range(h):
            next_weight = RBF_weights[coord[0], h] * RBF_weights[coord[1], j]
            weighted_sum += r2z(C_sub[h, j]) * next_weight
            sum_of_weights += next_weight
    return z2r(weighted_sum / sum_of_weights)



def expand_matrix(output_list, R_full):
    """
    This function expands output from the pooled RBF-weighted averages at each coordinate (index in the full matrix)

    Parameters
    ----------
    output_list : list
        results from pooled expand_corrmat

    R_full : ndarray
        Full list of coordinates that pass kurtosis threshold

    Returns
    ----------
    results : ndarray
        Expanded full matrix (len(R_full) x len(R_full))

    """
    ### convert output list to array
    output_array = np.array(output_list)
    ### initialize a full matrix (len(R_full) x len(R_full))
    C_full = np.zeros([R_full.shape[0], R_full.shape[0]])
    ### find indices of for the top triangle in the full matrix - use those indices to fill in the correponding values from output array to full matrix
    C_full[np.tril_indices(R_full.shape[0], -1)] = output_array
    ### expand to full matrix
    return C_full + C_full.T + np.eye(C_full.shape[0])


def chunker(iterable, chunksize):
    """
        Chunks longer sequence by regular interval

        Parameters
        ----------
        iterable : list or ndarray
            Use would be a long timeseries that needs to be broken down

        chunksize : int
            Size to break down

        Returns
        ----------
        results : ndarray
            Chunked timeseries

        """
    return map(None, *[iter(iterable)] * chunksize)


def compile_corrs(path_to_npz_data, Corr_timeseries):
    """
        Compiles correlation values - as well as other subject/electrode specific paramters - creates the compiled pandas dataframe used for figures

        Parameters
        ----------
        path_to_npz_data : string
            Path to npz files - I know this isn't a great way to do this :/

        Corr_timeseries : npz file
            npz file containing correlation values (loop outside - for each electrode)

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode, correlation, samples, and sample rate

        """
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
    ### parse path is necessary for the wacky naming system
    f_name, electrode = parse_path_name(Corr_timeseries)
    corr_data = np.load(Corr_timeseries, mmap_mode='r')
    npz_data = np.load(os.path.join(path_to_npz_data, f_name + '.npz'), mmap_mode='r')
    tempR = round_it(corr_data['coord'], 2)
    tempmeancorr = z2r(np.mean(r2z(corr_data['corrs'])))
    tempsamplerate = np.mean(npz_data['samplerate'])
    tempsamples = np.shape(npz_data['Y'])[0]

    return pd.DataFrame({'R': [tempR], 'Correlation': [tempmeancorr], 'Subject': [f_name], 'Electrode': [electrode], 'Sample rate' : [tempsamplerate], 'Samples': [tempsamples]})

def compile_nn_corrs(nn_corr_file):
    """
        Compiles correlation values from nearest neighbor - as well as other subject and electrode location

        Parameters
        ----------

        Corr_timeseries : npz file
            npz file containing correlation values for nearest neighbor (loop outside - for each electrode)

        Returns
        ----------
        results : dataframe
            compiled dataframe with: Subject, electrode location, and correlation

        """
    def parse_path_name(path_name):
        if os.path.basename(path_name).count('_') == 2:
            f_name = os.path.splitext(os.path.basename(path_name))[0].split("_", 2)[0]
            return f_name
        elif os.path.basename(path_name).count('_') == 3:
            f_name = '_'.join(os.path.splitext(os.path.basename(path_name))[0].split("_", 3)[0:2])
            return f_name
        else:
            return "error"
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


def density(n_by_3_Locs, nearest_n):
    """
        Calculates the density of the nearest n neighbors

        Parameters
        ----------

        n_by_3_Locs : ndarray
            Array of electrode locations - one for each row

        nearest_n : int
            Number of nearest neighbors to consider in density calculation

        Returns
        ----------
        results : ndarray
            Denisity for each electrode location

        """
    nbrs = NearestNeighbors(n_neighbors=nearest_n, algorithm='ball_tree').fit(n_by_3_Locs)
    distances, indices = nbrs.kneighbors(n_by_3_Locs)
    return np.exp(-(distances.sum(axis=1) / (np.shape(distances)[1] - 1)))


def normalize_Y(Y_matrix):
    """
         Normalizes timeseries

         Parameters
         ----------

         Y_matrix : ndarray
             Raw activity from each electrode channel

         Returns
         ----------
         results : ndarray
             Normalized activity from each electrode channel

         """
    Y = Y_matrix
    m = mat.repmat(np.min(Y, axis = 0), Y.shape[0], 1)
    Y = Y - m
    m = mat.repmat(np.max(Y, axis = 0), Y.shape[0], 1)
    Y = np.divide(Y,m)
    added = mat.repmat(0.5 + np.arange(Y.shape[1]), Y.shape[0], 1)
    Y = Y + added
    return pd.DataFrame(Y)

def compute_coord(coord, weights, Z):

    xweights = weights[coord[0], :]
    yweights = weights[coord[1], :]

    next_weights = np.outer(xweights, yweights)
    next_weights = next_weights - np.triu(next_weights)

    w = np.sum(next_weights)
    k = np.sum(Z * next_weights)

    return z2r(k / w)


def expand_corrmat_j(weights, C):
    n = weights.shape[0]
    K = np.zeros([n, n])
    W = np.zeros([n, n])

    Z = r2z(C)
    for x in range(n):
        xweights = weights[x, :]
        for y in range(x):
            yweights = weights[y, :]

            next_weights = np.outer(xweights, yweights)
            next_weights = next_weights - np.triu(next_weights)

            W[x, y] = np.sum(next_weights)
            K[x, y] = np.sum(Z * next_weights)

    return K + K.T, W + W.T


def expand_corrmat_parsed(weights, C, mode='fit'):
    n = weights.shape[0]
    s = C.shape[0]
    K = np.zeros([n, n])
    W = np.zeros([n, n])
    Z = r2z(C)

    predict_mode = (mode == 'predict')

    for x in range(n):
        xweights = weights[x, :]
        if predict_mode:
            vals = range(x, n)
        else:
            vals = range(x)
        for y in vals:
            if predict_mode and (y < (n - s)): #this may be off by one index
                continue
            yweights = weights[y, :]

            next_weights = np.outer(xweights, yweights)
            next_weights = next_weights - np.triu(next_weights)

            W[x, y] = np.sum(next_weights)
            K[x, y] = np.sum(Z * next_weights)

    return K + K.T, W + W.T