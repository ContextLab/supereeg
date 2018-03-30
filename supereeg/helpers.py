from __future__ import division
from __future__ import print_function

import multiprocessing
import copy
import os
import warnings
import numpy.matlib as mat
import pandas as pd
import numpy as np
import imageio
import nibabel as nib
import hypertools as hyp

from nilearn import plotting as ni_plt
from nilearn import image
from nilearn.input_data import NiftiMasker
from scipy.stats import kurtosis, zscore, pearsonr
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from scipy import linalg
from scipy.ndimage.interpolation import zoom
from joblib import Parallel, delayed


def _std(res=None):
    """
    Load a Nifti image of the standard MNI 152 brain at the given resolution


    Parameters
    ----------
    res : int or float or None
        If int or float: (for cubic voxels) or a list or array of 3D voxel dimensions
        If None, returns loaded gray matter masked brain

    Returns
    ----------
    results : Nifti1Image
         Nifti image of the standard brain

    """
    from .nifti import Nifti
    from .load import load

    std_img = load('std')
    if res:
        return _resample_nii(std_img, res)
    else:
        return std_img


def _gray(res=None):
    """
    Load a Nifti image of the gray matter masked MNI 152 brain at the given resolution


    Parameters
    ----------
    res : int or float or None
        If int or float: (for cubic voxels) or a list or array of 3D voxel dimensions
        If None, returns loaded gray matter masked brain

    Returns
    ----------
    results : Nifti1Image
         Nifti image of gray masked brain

    """
    from .nifti import Nifti
    from .load import load

    gray_img = load('gray')
    threshold = 100
    gray_data = gray_img.get_data()
    gray_data[np.isnan(gray_data) | (gray_data < threshold)] = 0

    if np.iterable(res) or np.isscalar(res):
        return _resample_nii(Nifti(gray_data, gray_img.affine), res)
    else:
        return Nifti(gray_data, gray_img.affine)


def _resample_nii(x, target_res, precision=5):
    """
    Resample a Nifti image to have the given voxel dimensions


    Parameters
    ----------
    x : Nifti1Image
        Input Nifti image (a nibel Nifti1Image object)

    target_res : int or float or None
        Int or float (for cubic voxels) or a list or array of 3D voxel dimensions

    precision : int
        Number of decimal places in affine transformation matrix for resulting image (default: 5)

    Returns
    ----------
    results : Nifti1Image
         Re-scaled Nifti image

    """

    from .nifti import Nifti

    if np.any(np.isnan(x.get_data())):
        img = x.get_data()
        img[np.isnan(img)] = 0.0
        x = nib.nifti1.Nifti1Image(img, x.affine)

    res = x.header.get_zooms()[0:3]
    scale = np.divide(res, target_res).ravel()

    target_affine = x.affine

    target_affine[0:3, 0:3] /= scale
    target_affine = np.round(target_affine, decimals=precision)

    # correct for 1-voxel shift
    target_affine[0:3, 3] -= np.squeeze(np.multiply(np.divide(target_res, 2.0), np.sign(target_affine[0:3, 3])))
    target_affine[0:3, 3] += np.squeeze(np.sign(target_affine[0:3, 3]))

    if len(scale) < np.ndim(x.get_data()):
        assert np.ndim(x.get_data()) == 4, 'Data must be 3D or 4D'
        scale = np.append(scale, x.shape[3])

    z = zoom(x.get_data(), scale)
    try:
        z[z < 1e-5] = np.nan
    except:
        pass
    return Nifti(z, target_affine)


def _apply_by_file_index(bo, xform, aggregator):
    """
    Session dependent function application and aggregation

    Parameters
    ----------
    bo : Brain object
        Contains data

    xform : function
        The function to apply to the data matrix from each filename

    aggregator: function
        Function for aggregating results across multiple iterations

    Returns
    ----------
    results : numpy ndarray
         Array of aggregated results

    """

    for idx, session in enumerate(bo.sessions.unique()):
        if idx is 0:
            results = xform(bo.get_data().as_matrix()[bo.sessions == session, :])
        else:
            results = aggregator(results, xform(bo.get_data().as_matrix()[bo.sessions == session, :]))

    return results


def _kurt_vals(bo):
    """
    Function that calculates maximum kurtosis values for each channel

    Parameters
    ----------
    bo : Brain object
        Contains data

    Returns
    ----------
    results: 1D ndarray
        Maximum kurtosis across sessions for each channel

    """
    sessions = bo.sessions.unique()
    results = list(map(lambda s: kurtosis(bo.data[(s==bo.sessions).values]), sessions))
    return np.max(np.vstack(results), axis=0)

    # def aggregate(prev, next):
    #     return np.max(np.vstack((prev, next)), axis=0)
    #
    # return _apply_by_file_index(bo, kurtosis, aggregate)


def _get_corrmat(bo):
    """
    Function that calculates the average subject level correlation matrix for brain object across session

    Parameters
    ----------
    bo : Brain object
        Contains data


    Returns
    ----------
    results: 2D np.ndarray
        The average correlation matrix across sessions

    """

    def aggregate(p, n):
        return p + n

    def zcorr(x):
        return _r2z(1 - squareform(pdist(x.T, 'correlation')))

    summed_zcorrs = _apply_by_file_index(bo, zcorr, aggregate)

    return _z2r(summed_zcorrs / len(bo.sessions.unique()))


def _z_score(bo):
    """
    Function that calculates the average subject level correlation matrix for brain object across session

    Parameters
    ----------
    bo : Brain object
        Contains data

    Returns
    ----------
    results: 2D np.ndarray
        The average correlation matrix across sessions

    """

    def aggregate(p, n):
        return np.vstack((p, n))

    def z(x):
        return zscore(x)

    z_scored_data= _apply_by_file_index(bo, z, aggregate)

    return z_scored_data



def _z2r(z):
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
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def _r2z(r):
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
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


def _rbf(x, center, width=20):
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
    results : ndarray
        Matrix of _rbf weights for each subject coordinate for all coordinates

    """
    return np.exp(-cdist(x, center, metric='euclidean') ** 2 / float(width))


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
    tmp = inpoints[2, :] < 0
    inpoints[:, tmp] = linalg.solve(np.dot(rotmat, down), inpoints[:, tmp])
    inpoints[:, ~tmp] = linalg.solve(np.dot(rotmat, up), inpoints[:, ~tmp])

    return _round_it(inpoints[0:3, :].T, 2)


def _uniquerows(x):
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


def _expand_corrmat_fit(C, weights):
    """
    Gets full correlation matrix

    Parameters
    ----------
    C : Numpy array
        Subject's correlation matrix

    weights : Numpy array
        Weights matrix calculated using _rbf function matrix

    mode : str
        Specifies whether to compute over all elecs (fit mode) or just new elecs
        (predict mode)

    Returns
    ----------
    numerator : Numpy array
        Numerator for the expanded correlation matrix
    denominator : Numpy array
        Denominator for the expanded correlation matrix
    """

    C[np.eye(C.shape[0]) == 1] = 0
    C[np.where(np.isnan(C))] = 0

    n = weights.shape[0]
    K = np.zeros([n, n])
    W = np.zeros([n, n])
    Z = C

    s = 0

    vals = list(range(s, n))
    for x in vals:
        xweights = weights[x, :]

        vals = list(range(x))
        for y in vals:
            yweights = weights[y, :]

            next_weights = np.outer(xweights, yweights)
            next_weights = next_weights - np.triu(next_weights)

            W[x, y] = np.sum(next_weights)
            K[x, y] = np.sum(Z * next_weights)
    return (K + K.T), (W + W.T)


def _expand_corrmat_predict(C, weights):
    """
    Gets full correlation matrix

    Parameters
    ----------
    C : Numpy array
        Subject's correlation matrix

    weights : Numpy array
        Weights matrix calculated using _rbf function matrix

    mode : str
        Specifies whether to compute over all elecs (fit mode) or just new elecs
        (predict mode)

    Returns
    ----------
    numerator : Numpy array
        Numerator for the expanded correlation matrix
    denominator : Numpy array
        Denominator for the expanded correlation matrix

    """

    C[np.eye(C.shape[0]) == 1] = 0
    C[np.where(np.isnan(C))] = 0

    n = weights.shape[0]
    K = np.zeros([n, n])
    W = np.zeros([n, n])
    Z = C

    s = C.shape[0]
    sliced_up = [(x, y) for x in range(s, n) for y in range(x)]

    results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(_compute_coord)(coord, weights, Z) for coord in sliced_up)

    W[[x[0] for x in sliced_up], [x[1] for x in sliced_up]] = [x[0] for x in results]
    K[[x[0] for x in sliced_up], [x[1] for x in sliced_up]] = [x[1] for x in results]

    return (K + K.T), (W + W.T)


def _compute_coord(coord, weights, Z):
    next_weights = np.outer(weights[coord[0], :], weights[coord[1], :])
    next_weights = next_weights - np.triu(next_weights)
    return np.sum(next_weights), np.sum(Z * next_weights)


def _chunk_bo(bo, chunk):
    """
    Chunk brain object by session for reconstruction. Returns chunked indices

        Parameters
    ----------
    bo : brain object
        Brain object used to reconstruct and data to chunk

    chunk : list
        Chunked indices


    Returns
    ----------
    nbo : brain object
        Chunked brain object with chunked zscored data in the data field

    """
    return bo.get_slice(sample_inds=[i for i in chunk if i is not None])


def _timeseries_recon(bo, K, chunk_size=1000, preprocess='zscore'):
    """
    Reconstruction done by chunking by session

        Parameters
    ----------
    bo : Brain object
        Data to be reconstructed

    K : Numpy.ndarray
        Correlation matix including observed and predicted locations

    chunk_size : int
        Size to break data into

    Returns
    ----------
    results : ndarray
        Compiled reconstructed timeseries

    """
    if preprocess==None:
        data = bo.get_data().as_matrix()
    elif preprocess=='zscore':
        if bo.data.shape[0]<3:
            warnings.warn('Not enough samples to zscore so it will be skipped.'
            ' Note that this will cause problems if your data are not already '
            'zscored.')
            data = bo.get_data().as_matrix()
        else:
            data = bo.get_zscore_data()

    s = K.shape[0] - data.shape[1]
    Kba = K[:s, s:]
    Kaa = K[s:, s:]
    Kaa_inv = np.linalg.pinv(Kaa)
    sessions = bo.sessions.unique()
    chunks = [np.array(i) for session in sessions for i in _chunker(bo.sessions[bo.sessions == session].index.tolist(), chunk_size)]
    chunks = list(map(lambda x: np.array(x[x != np.array(None)], dtype=np.int8), chunks))
    # results = np.vstack(Parallel(n_jobs=multiprocessing.cpu_count())(
    #     delayed(_reconstruct_activity)(data[chunk, :], Kba, Kaa_inv) for chunk in chunks))
    results = np.vstack(list(map(lambda x: _reconstruct_activity(data[x, :], Kba, Kaa_inv), chunks)))
    zresults = list(map(lambda s: zscore(results[bo.sessions==s, :]), sessions))
    return np.hstack([np.vstack(zresults), data])

def _chunker(iterable, chunksize, fillvalue=None):
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
    try:
        from itertools import zip_longest as zip_longest
    except:
        from itertools import izip_longest as zip_longest

    args = [iter(iterable)] * chunksize
    return list(zip_longest(*args, fillvalue=fillvalue))


def _reconstruct_activity(Y, Kba, Kaa_inv):
    """
    Reconstruct activity

    Parameters
    ----------
    Y : numpy array
        brain object with zscored data

    Kba : correlation matrix (unknown to known)

    Kaa_inv : inverse correlation matrix (known to known)

    zscore = False

    Returns
    ----------
    results : ndarray
        Reconstructed timeseries

    """
    return np.atleast_2d(np.squeeze(np.dot(np.dot(Kba, Kaa_inv), Y.T).T))

def _round_it(locs, places): #TODO: do we need a separate function for this?  doesn't seem much more convenient than the np.round function...
    """
    Rounding function

    Parameters
    ----------
    locs : array or float
        Number be rounded

    places : int
        Number of places to round

    Returns
    ----------
    result : array or float
        Rounded number

    """
    return np.round(locs, decimals=places)


def filter_elecs(bo, measure='kurtosis', threshold=10):
    """
    Filter electrodes based on kurtosis value

    Parameters
    ----------
    bo : brain object
        Brain object

    measure : 'kurtosis'
        Method to filter electrodes. Only kurtosis supported currently.

    threshold : int
        Threshold for filtering

    Returns
    ----------
    result : brain object
        Brain object with electrodes and corresponding data that passes kurtosis thresholding

    """
    thresh_bool = bo.kurtosis > threshold
    nbo = copy.deepcopy(bo) #TODO: modify bo.get_locs rather than copying brain object again here
    nbo.data = bo.data.loc[:, ~thresh_bool]
    nbo.locs = bo.locs.loc[~thresh_bool]
    nbo.n_elecs = bo.data.shape[1]
    return nbo


def filter_subj(bo, measure='kurtosis', return_locs=False, threshold=10):
    """
    Filter subjects if less than two electrodes pass kurtosis value

    Parameters
    ----------
    bo : str
        Path to brain object

    measure : 'kurtosis'
        Method to filter electrodes. Only kurtosis supported currently.

    return_locs : bool
        Default False, returns meta data. If True, returns electrode locations that pass kurtosis threshold

    threshold : int
        Threshold for filtering.

    Returns
    ----------
    result : meta brain object or None
        Meta field from brain object if two or more electrodes pass kurtosis thresholding.

    """
    from .load import load

    locs = load(bo, field='locs')
    kurt_vals = load(bo, field='kurtosis')
    meta = load(bo, field='meta')

    if not meta is None:
        thresh_bool = kurt_vals > threshold
        if sum(~thresh_bool) < 2:
            print(meta + ': not enough electrodes pass threshold')

        else:
            if return_locs:
                locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
                return meta, locs[~thresh_bool]
            else:
                return meta
    else:
        print('no meta data for brain object')


def _corr_column(X, Y):
    return np.array([pearsonr(x, y)[0] for x, y in zip(X.T, Y.T)])


def _normalize_Y(Y_matrix): #TODO: should be part of bo.get_data and/or Brain.__init__
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
    m = mat.repmat(np.min(Y, axis=0), Y.shape[0], 1)
    Y = Y - m
    m = mat.repmat(np.max(Y, axis=0), Y.shape[0], 1)
    Y = np.divide(Y, m)
    added = mat.repmat(0.5 + np.arange(Y.shape[1]), Y.shape[0], 1)
    Y = Y + added
    return pd.DataFrame(Y)


def _fullfact(dims):
    '''
    Replicates MATLAB's _fullfact function (behaves the same way)
    '''
    vals = np.asmatrix(list(range(1, dims[0] + 1))).T
    if len(dims) == 1:
        return vals
    else:
        aftervals = np.asmatrix(_fullfact(dims[1:]))
        inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
        row = 0
        for i in range(aftervals.shape[0]):
            inds[row:(row + len(vals)), 0] = vals
            inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
            row += len(vals)
        return inds


def model_compile(data):
    """
    Compile existing expanded correlation matrices.

    Parameters
    ----------
    data : list of model object file directories
        Compiles model objects

    Returns
    ----------
    model : Model object
        A new updated model object

    """
    from .load import load
    from .model import Model

    m = load(data[0])
    numerator = m.numerator
    denominator = m.denominator
    n_subs = 1

    for mo in data[1:]:
        m = load(mo)
        # numerator = np.nansum(np.dstack((numerator, m.numerator)), 2)
        numerator += m.numerator
        denominator += m.denominator
        n_subs += 1

    return Model(numerator=numerator, denominator=denominator,
                 locs=m.locs, n_subs=n_subs)
    ### this concatenation of locations doesn't work when updating an existing model (but would be necessary for a build)
    # return Model(numerator=numerator, denominator=denominator,
    #              locs=pd.concat([m.locs, bo.locs]), n_subs=n_subs)


def _near_neighbor(bo, mo, match_threshold='auto'): #TODO: should this be part of bo.get_locs() or Brain.__init__, or possibly model.__init__?
    """
    Finds the nearest voxel for each subject's electrode location and uses
    that as revised electrodes location matrix in the prediction.

    Parameters
    ----------

    bo : Brain object
        Brain object to update

    mo : Model object
        Model object for the nearests locations used to predict

    match_threshold : 'auto', int, or None
        Threshold used to find nearest neighbor

        options:

        match_threshold = 'auto' : include only nearest neighbor if falls within one voxel distance

        match_threshold =  0 :set nearest_neighbor = False and proceed (only exact matches will be used)

        match_threshol = None use best match and don't check (even if far away)

        match_threshold > 0 : include only nearest neighbor that are within given distance

    Returns
    ----------
    bo : Brain object
        A new updated brain object

    """

    nbo = copy.deepcopy(bo) #FIXME: copying is expensive...
    nbo.orig_locs = nbo.locs
    d = cdist(nbo.locs, mo.locs, metric='Euclidean')
    for i in range(len(nbo.locs)):
        min_ind = list(zip(*np.where(d == d.min())))[0]
        nbo.locs.iloc[min_ind[0], :] = mo.locs.iloc[min_ind[1], :]
        d[min_ind[0]] = np.inf
        d[:, min_ind[1]] = np.inf
    if not match_threshold is 0 or None:

        if match_threshold is 'auto':
            v_size = _vox_size(mo.locs)
            thresh_bool = abs(nbo.locs - bo.locs) > v_size
            thresh_bool = thresh_bool.any(1).ravel()
        else:
            thresh_bool = abs(nbo.locs - bo.locs) > match_threshold
            thresh_bool = thresh_bool.any(1).ravel()
            assert match_threshold > 0, 'Negative Euclidean distances are not allowed'
        nbo.data = nbo.data.loc[:, ~thresh_bool]
        nbo.locs = nbo.locs.loc[~thresh_bool, :]
        nbo.n_elecs = nbo.data.shape[1]
        nbo.kurtosis = nbo.kurtosis[~thresh_bool]
        return nbo
    else:
        return nbo


def _vox_size(locs):
    """
    Finds voxel size

    Parameters
    ----------
    locs : pandas DataFrame
        Locations in brain extracted from nifti

    Returns
    ----------
    results : ndarray
        1 x n_dims of voxel size

    """
    from .brain import Brain
    bo_n = Brain(data=np.array([0]))
    n_dims = locs.shape[1]
    v_size = np.zeros([1, n_dims])
    # make voxel function
    for i in np.arange(n_dims):
        a = np.unique(locs.iloc[:, i])
        dists = pdist(np.atleast_2d(a).T, 'euclidean')
        #v_size[0][i] = np.min(dists[dists > 0])
        if np.sum(dists > 0) > 0:
            v_size[0][i] = np.min(dists[dists > 0])
        else:
            v_size[0][i] = bo_n.minimum_voxel_size
    return v_size


def sort_unique_locs(locs):
    """
    Sorts unique locations

    Parameters
    ----------
    locs : pandas DataFrame or ndarray
        Electrode locations

    Returns
    ----------
    results : ndarray
        Array of unique locations

    """
    if isinstance(locs, pd.DataFrame):
        unique_full_locs = np.vstack(set(map(tuple, locs.as_matrix())))
    elif isinstance(locs, np.ndarray):
        unique_full_locs = np.vstack(set(map(tuple, locs)))
    else:
        print('unknown location type')

    return unique_full_locs[unique_full_locs[:, 0].argsort(),]


def _count_overlapping(X, Y):
    """
    Finds overlapping locations (Y in X)

    Parameters
    ----------
    X : brain object or model object
        Electrode locations

    Y : brain object or model object
        Electrode locations

    Returns
    ----------
    results : ndarray
        Array of length(X.locs) with 0s and 1s, where 1s denote overlapping locations Y in X

    """

    return np.sum([(X.locs == y).all(1) for idy, y in Y.locs.iterrows()], 0).astype(bool)


def make_gif_pngs(nifti, gif_path, index=range(100, 200), name=None, **kwargs):
    """
    Plots series of nifti timepoints as nilearn plot_glass_brain in .png format

    Parameters
    ----------
    nifti : nib.nifti1.Nifti1Image
        Nifti of reconconstruction

    gif_path : directory
        Directory to save .png files

    name : str
        Name for gif, default is None and will name the file based on the time windows

    window_min : int
        Lower bound for time window.

    window_max : int
        Upper bound for time window.

    Returns
    ----------
    results : png
        Series of pngs

    """

    for i in index:
        nii_i = image.index_img(nifti, i)
        outfile = os.path.join(gif_path, str(i).zfill(4) + '.png')
        ni_plt.plot_glass_brain(nii_i, output_file=outfile, **kwargs)

    images = []
    for file in os.listdir(gif_path):
        if file.endswith(".png"):
            images.append(imageio.imread(os.path.join(gif_path, file)))
    if name is None:
        gif_outfile = os.path.join(gif_path, 'gif_' + str(min(index)) + '_' + str(max(index)) + '.gif')

    else:
        gif_outfile = os.path.join(gif_path, str(name) + '.gif')
    imageio.mimsave(gif_outfile, images)


def _data_and_samplerate_by_file_index(bo, xform, **kwargs):
    """
    Session dependent function application and aggregation

    Parameters
    ----------
    bo : Brain object
        Contains data

    xform : function
        The function to apply to the data matrix from each filename

    aggregator: function
        Function for aggregating results across multiple iterations

    Returns
    ----------
    results : numpy ndarray
         Array of aggregated results

    """
    sample_rate = []

    for idx, session in enumerate(bo.sessions.unique()):
        if idx is 0:
            data_results, session_results, sr_results = xform(bo.data.loc[bo.sessions == session],
                                                                       bo.sessions.loc[bo.sessions == session],
                                                                       bo.sample_rate[idx], **kwargs)
            sample_rate.append(sr_results)
        else:
            data_next, session_next, sr_next = xform(bo.data.loc[bo.sessions == session, :],
                                                                           bo.sessions.loc[bo.sessions == session],
                                                                           bo.sample_rate[idx], **kwargs)
            data_results = data_results.append(data_next, ignore_index=True)
            session_results = session_results.append(session_next, ignore_index=True)
            sample_rate.append(sr_next)

    return data_results, session_results, sample_rate



def _resample(bo, resample_rate=64):
    """
    Function that resamples data to specified sample rate

    Parameters
    ----------
    bo : Brain object
        Contains data

    Returns
    ----------
    results: 2D np.ndarray
        Resampled data - pd.DataFrame
        Resampled sessions - pd.DataFrame
        Resample rate - List

    """


    def _resamp(data, session, sample_rate, resample_rate):

        # number of samples for resample
        n_samples = np.round(np.shape(data)[0] * resample_rate / sample_rate)

        # index for those samples
        resample_index = np.round(np.linspace(data.index.min(), data.index.max(), n_samples))

        # resampled sessions
        re_session = session[resample_index]
        re_session.interpolate(method='pchip', inplace=True, limit_direction='both')

        # resampled data
        re_data = data.loc[resample_index]
        re_data.interpolate(method='pchip', inplace=True, limit_direction='both')

        return re_data, re_session, resample_rate

    return _data_and_samplerate_by_file_index(bo, _resamp, resample_rate=resample_rate)


def _plot_locs_connectome(locs, label=None, pdfpath=None):
    """
    Plots locations in nilearn plot connectome

    Parameters
    ----------
    locs : pd.DataFrame
        Electrode locations

    Returns
    ----------
    results: nilearn connectome plot
        plot of electrodes


    """
    if label is not None:
        label = list(map(lambda x: [0,0,0] if x=='observed' else [1,0,0], label))
        colors = np.asarray(label)
        colors = list(map(lambda x: x[0], np.array_split(colors, colors.shape[0], axis=0)))
    else:
        colors = 'k'
    ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=pdfpath,
                           node_kwargs={'alpha': 0.5, 'edgecolors': None},
                           node_size=10, node_color=colors)
    if not pdfpath:
        ni_plt.show()

def _plot_locs_hyp(locs, pdfpath): #TODO: do we need a separate function for this?  doesn't look more convenient than calling hyp.plot directly...

    """
    Plots locations in hypertools

    Parameters
    ----------
    locs : pd.DataFrame
        Electrode locations

    Returns
    ----------
    results: nilearn connectome plot
        plot of electrodes


    """
    hyp.plot(locs, 'k.', save_path=pdfpath)

def _plot_glass_brain(nifti, pdfpath, index=1): #TODO: do we need a separate function for this?  doesn't look more convenient than calling plot_glas_brain directly...
    """
    Plots nifti data

    Parameters
    ----------
    nifti : nifti image
        Nifti image to plot

    Returns
    ----------
    results: nilearn plot_glass_brain
        plot data


    """
    nii = image.index_img(nifti, index)
    ni_plt.plot_glass_brain(nii)
    if not pdfpath:
        ni_plt.show()

def _nifti_to_brain(nifti, mask_file=None):

    """
    Takes or loads nifti file and converts to brain object

    Parameters
    ----------
    nifti : str or nifti image

        If nifti is a nifti filepath, loads nifti and returns brain object

        If nifti is a nifti image, it returns a brain object

    Returns
    ----------
    results: brain object


    """
    from .nifti import Nifti

    if type(nifti) is Nifti:
        img = nifti

    elif type(nifti) is nib.nifti1.Nifti1Image:
        img = nifti

    elif type(nifti) is str:
        if os.path.exists(nifti):
            img = nib.load(nifti)
        else:
            warnings.warn('Nifti format not supported')
    else:
        warnings.warn('Nifti format not supported')

    mask = NiftiMasker(mask_strategy='background')
    if mask_file is None:
        mask.fit(nifti)
    else:
        mask.fit(mask_file)

    hdr = img.header
    S = img.get_sform()

    Y = np.float64(mask.transform(nifti)).copy()
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = _fullfact(img.shape[0:3])[vmask, ::-1] - 1

    R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]

    return Y, R, {'header': hdr}


def _brain_to_nifti(bo, nii_template): #FIXME: this is incredibly inefficient; could be done much faster using reshape and/or nilearn masking

    """
    Takes or loads nifti file and converts to brain object

    Parameters
    ----------
    bo : brain object

    template : str, Nifti1Image, or None

        Template is a nifti file with the desired resolution to save the brain object activity


    Returns
    ----------
    results: nibabel.Nifti1Image
        A nibabel nifti image


    """
    from .nifti import Nifti

    hdr = nii_template.get_header()
    temp_v_size = hdr.get_zooms()[0:3]

    R = bo.get_locs()
    Y = bo.data.as_matrix()
    Y = np.array(Y, ndmin=2)
    S = nii_template.affine
    locs = np.array(np.dot(R - S[:3, 3], np.linalg.inv(S[0:3, 0:3])), dtype='int')

    shape = np.max(np.vstack([np.max(locs, axis=0) + 1, nii_template.shape[0:3]]), axis=0)
    data = np.zeros(tuple(list(shape) + [Y.shape[0]]))
    counts = np.zeros(data.shape)

    for i in range(R.shape[0]):
        data[locs[i, 0], locs[i, 1], locs[i, 2], :] += Y[:, i]
        counts[locs[i, 0], locs[i, 1], locs[i, 2], :] += 1

    with np.errstate(invalid='ignore'):
        for i in range(R.shape[0]):
            data[locs[i, 0], locs[i, 1], locs[i, 2], :] = np.divide(data[locs[i, 0], locs[i, 1], locs[i, 2], :], counts[locs[i, 0], locs[i, 1], locs[i, 2], :])

    return Nifti(data, affine=nii_template.affine)
