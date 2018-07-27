from __future__ import division
from __future__ import print_function

#TODO: there are multiple implementations of functions like _apply_by_file_index.  these should be consolidated into one
#common function that is used and called multiple times.  In addition, aggregator and transform functions that are used
#across apply_by_file wrappers should be shared (rather than defined multiple times).  We could also call_apply_by_file_index
#"groupby" to conform to the pandas style.  e.g. bo.groupby(session) returns a generator whose produces are brain objects
#each of one session.  we could then use bo.groupby(session).aggregate(xform) to produce a list of objects, where each is
#comprised of the xform applied to the brain object containing one session worth of data from the original object.

import copy
import os
import numpy.matlib as mat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import nibabel as nib
import hypertools as hyp
import shutil
import warnings


from nilearn import plotting as ni_plt
from nilearn import image
from nilearn.input_data import NiftiMasker
from scipy.stats import kurtosis, zscore, pearsonr
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from scipy.special import logsumexp
from scipy import linalg
from scipy.ndimage.interpolation import zoom
try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest


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
        session_xform = xform(bo.get_slice(sample_inds=np.where(bo.sessions == session)[0], inplace=False))
        if idx is 0:
            results = session_xform
        else:
            results = aggregator(results, session_xform)

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

    def zcorr_xform(bo):
        return np.multiply(bo.dur, _r2z(1 - squareform(pdist(bo.get_data().T, 'correlation'))))

    summed_zcorrs = _apply_by_file_index(bo, zcorr_xform, aggregate)

    #weight each session by recording time
    return _z2r(summed_zcorrs / np.sum(bo.dur))


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
    def z_score_xform(bo):
        return zscore(bo.get_data())

    def vstack_aggregrate(x1, x2):
        return np.vstack((x1, x2))

    return _apply_by_file_index(bo, z_score_xform, vstack_aggregrate)



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
    warnings.simplefilter('ignore')
    if isinstance(z, list):
        z = np.array(z)
    r = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    if isinstance(r, np.ndarray):
        r[np.isinf(z) & (z > 0)] = 1
        r[np.isinf(z) & (z < 0)] = -1
    else:
        if np.isinf(z) & (z > 0):
            return 1
        elif np.isinf(z) & (z < 0):
            return -1
    return r

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
    warnings.simplefilter('ignore')
    return 0.5 * (np.log(1 + r) - np.log(1 - r))


def _log_rbf(to_coords, from_coords, width=20):
    """
    Radial basis function

    Parameters
    ----------
    to_coords : ndarray
        Series of all coordinates (one per row) - R_full

    c : ndarray
        Series of subject's coordinates (one per row) - R_subj

    width : positive scalar
        Radius

    Returns
    ----------
    results : ndarray
        Matrix of log rbf weights for each subject coordinate for all coordinates

    """
    assert np.isscalar(width), 'RBF width must be a scalar'
    assert width > 0, 'RBF width must be positive'
    weights = -cdist(to_coords, from_coords, metric='euclidean') ** 2 / float(width)
    return weights


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

    return np.round(inpoints[0:3, :].T, decimals=2)


def _blur_corrmat(Z, weights):
    """
    Gets full correlation matrix

    Parameters
    ----------
    Z : Numpy array
        Subject's Fisher z-transformed correlation matrix

    weights : Numpy array
        Weights matrix calculated using _log_rbf function matrix

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
    #import seaborn as sns
    #import matplotlib.pyplot as plt

    triu_inds = np.triu_indices(Z.shape[0], k=1)

    #need to do computations seperately for positive and negative values
    sign_Z_full = np.sign(Z)
    #logZ_pos_full = np.log(np.multiply(sign_Z_full > 0, Z))
    #logZ_neg_full = np.log(np.multiply(sign_Z_full < 0, np.abs(Z)))

    sign_Z = sign_Z_full[triu_inds]
    logZ_pos = np.log(np.multiply(sign_Z > 0, Z[triu_inds]))
    logZ_neg = np.log(np.multiply(sign_Z < 0, np.abs(Z[triu_inds])))

    n = weights.shape[0]
    K_pos = np.zeros([n, n])
    K_neg = np.zeros([n, n])
    W = np.zeros([n, n])

    for x in range(n-1):
        xweights = weights[x, :]
        x_match = np.isclose(xweights, 0)
        for y in range(x+1, n): #fill in upper triangle only
            yweights = weights[y, :]
            y_match = np.isclose(yweights, 0)

            if np.any(x_match) and np.any(y_match): #the pair of locations we're filling in already exists in the given data
                x_ind = np.where(x_match)[0]
                y_ind = np.where(y_match)[0]
                Z_match_val = np.mean(Z[x_ind, y_ind])
                W[x, y] = 0.
                if Z_match_val > 0:
                    K_pos[x, y] = np.log(Z_match_val)
                    K_neg[x, y] = -np.inf
                else:
                    K_pos[x, y] = -np.inf
                    K_neg[x, y] = np.log(np.abs(Z_match_val))
                continue

            next_weights = np.add.outer(xweights, yweights)
            next_weights = next_weights[triu_inds]

            W[x, y] = logsumexp(next_weights)
            K_pos[x, y] = logsumexp(logZ_pos + next_weights)
            K_neg[x, y] = logsumexp(logZ_neg + next_weights)

    #turn K_neg into complex numbers.  Where K_neg is infinite, this results in nans for the real number parts, so we'll
    #set any nans in K_neg.real to 0
    #TODO: the next lines are redundant with code in _to_log_complex; consolidate
    K_neg = np.multiply(0+1j, K_neg)
    K_neg.real[np.isnan(K_neg)] = 0
    K = K_pos + K_neg

    return K + K.T, W + W.T

def _to_log_complex(X):
    """
    Compute the log of the given numpy array.  Store all positive members of the original array in the real component of
    the result and all negative members of the original array in the complex component of the result.

    Parameters
    ----------
    X : numpy array to take the log of

    Returns
    ----------
    log_X_complex : The log of X, stored as complex numbers to keep track of the positive and negative parts
    """
    warnings.simplefilter('ignore')

    signX = np.sign(X)

    posX = np.log(np.multiply(signX > 0, X))
    posX[np.isnan(posX)] = 0

    negX = np.log(np.abs(np.multiply(signX < 0, X)))
    negX = np.multiply(0 + 1j, negX)
    negX.real[np.isnan(negX.real)] = 0

    return posX + negX

def _to_exp_real(C):
    """
    Inverse of _to_log_complex
    """
    posX = np.exp(C.real)
    if np.any(np.iscomplex(C)):
        negX = np.exp(C.imag)
        return posX - negX
    else:
        return posX


def _logsubexp(x,y):
    """
    Subtracts logged arrays
    Parameters
    ----------
    x : Numpy array
        Log complex array
    y : Numpy array
        Log complex array
    Returns
    ----------
    z : Numpy array
        Returns log complex array of x-y
    """
    if np.any(np.iscomplex(y)):
        y = _to_exp_real(y)
    else:
        y = np.exp(y)
    sub_log = _to_log_complex(x)
    neg_y_log = _to_log_complex(-y)
    sub_log.real = np.logaddexp(x.real, neg_y_log.real)
    sub_log.imag = np.logaddexp(x.imag, neg_y_log.imag)
    return sub_log


def _fill_upper_triangle(M, value):
    upper_tri = np.copy(M)
    upper_tri[np.triu_indices(upper_tri.shape[0], 1)] = value
    np.fill_diagonal(upper_tri, value)
    return upper_tri


def _timeseries_recon(bo, mo, chunk_size=1000, preprocess='zscore'):
    """
    Reconstruction done by chunking by session
        Parameters
    ----------
    bo : Brain object
        Data to be reconstructed

    mo : Model object
        Model to base the reconstructions on


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
    else:
        raise('Unsupported preprocessing option: ' + preprocess)

    brain_locs_in_model = _count_overlapping(mo.get_locs(), bo.get_locs())
    model_locs_in_brain = _count_overlapping(bo.get_locs(), mo.get_locs())

    if np.all(model_locs_in_brain):
        #if the model contains all of the locations (or fewer) than what are in the brain object, no reconstructions
        #are needed
        return data[:, brain_locs_in_model]

    #otherwise, we'll need to do some work
    Z = mo.get_model(z_transform=True)
    if ~np.any(brain_locs_in_model):
        #if none of the brain locations are in the model, we need to blur out the model to match up with the
        # locations in the brain object
        ### isnt this bypassed in the set_locs??
        combined_locs = np.vstack((bo.get_locs(), mo.get_locs()))
        model_locs_in_brain = [False]*bo.get_locs().shape[0]
        model_locs_in_brain.extend([True]*mo.get_locs().shape[0])

        rbf_weights = _log_rbf(combined_locs, mo.get_locs())
        Z = _blur_corrmat(Z, rbf_weights)

    K = _z2r(Z)

    known_inds, unknown_inds = known_unknown(mo.get_locs().as_matrix(), bo.get_locs().as_matrix(),
                                             bo.get_locs().as_matrix())

    Kaa = K[known_inds, :][:, known_inds]
    Kaa_inv = np.linalg.pinv(Kaa)

    Kba = K[unknown_inds, :][:, known_inds]

    sessions = bo.sessions.unique()
    try_filter = []
    chunks = [np.array(i) for session in sessions for i in _chunker(bo.sessions[bo.sessions == session].index.tolist(), chunk_size)]
    for i in chunks:
        try_filter.append([x for x in i if x is not None])
    #predict unobserved brain activitity
    combined_data = np.zeros((data.shape[0], K.shape[0]), dtype=data.dtype)
    combined_data[:, unknown_inds] = np.vstack(list(map(lambda x: _reconstruct_activity(data[x, :], Kba, Kaa_inv), try_filter)))
    combined_data[:, known_inds] = data

    for s in sessions:
        combined_data[bo.sessions==s, :] = zscore(combined_data[bo.sessions==s, :])

    return combined_data

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
    return np.dot(np.dot(Kba, Kaa_inv), Y.T).T


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

    m = load(data[0])
    m.update(data[1:])
    return m


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

def _unique(X):
    """
    Wrapper for np.unique and pd.unique that also returns matching indices

    Parameters
    ----------
    X : numpy array or pandas dataframe with electrode locations

    Returns
    ----------
    unique_X : the sorted unique rows of X

    unique_inds : the indices of X such that X[unique_inds, :] == X (or X.iloc[unique_inds] == X)
    """
    if X is None:
        return None, []

    dataframe = type(X) is pd.DataFrame
    if dataframe:
        columns = X.columns
        X = X.as_matrix()

    assert type(X) is np.ndarray, 'must pass in a numpy ndarray or dataframe'
    uX, inds = np.unique(X, axis=0, return_index=True)

    if dataframe:
        uX = pd.DataFrame(data=uX, columns=columns, index=np.arange(len(inds)))


    return uX, inds

def _union(X, Y): #TODO: add test for _union
    """
    Wrapper for np.vstack and pd.vstack that returns unique locations

    Parameters
    ----------
    X : numpy array or pandas dataframe with electrode locations

    Y : numpy array or pandas dataframe with electrode locations

    Returns
    ----------
    XY: the unique values of X and Y stacked together in the same format.  If either X or Y is a dataframe, then XY
        is a dataframe with the same columsn.  Otherwise XY is a numpy array.
    """

    if X is None:
        return Y
    elif Y is None:
        return X

    dataframeX = type(X) is pd.DataFrame
    if dataframeX:
        columnsX = X.columns
        X = X.as_matrix()

    dataframeY = type(Y) is pd.DataFrame
    if dataframeY:
        columnsY = Y.columns
        Y = Y.as_matrix()

    if dataframeX and dataframeY:
        assert np.all(columnsX == columnsY), 'Input dataframes have mismatched columns'

    assert X.shape[1] == Y.shape[1], 'Input data must have the same number of columns'

    XY = np.vstack((X, Y))
    XY_unique, tmp = _unique(XY)

    if dataframeX or dataframeY:
        if dataframeX:
            columns = columnsX
        else:
            columns = columnsY
        return pd.DataFrame(data=XY_unique, columns=columns, index=np.arange(XY_unique.shape[0]))
    else:
        return XY


def _empty(X): #TODO: ad test for _empty
    """
    Return true if X is None or if any element of X.shape is 0
    """

    if X is None:
        return True
    else:
        return np.any(np.isclose(X.shape, 0))


def get_rows(all_locations, subj_locations):
    """
        This function indexes a subject's electrode locations in the full array of electrode locations

        Parameters
        ----------
        all_locations : ndarray
            Full array of electrode locations

        subj_locations : ndarray
            Array of subject's electrode locations

        Returns
        ----------
        results : list
            Indexs for subject electrodes in the full array of electrodes

        """
    if subj_locations.ndim == 1:
        subj_locations = subj_locations.reshape(1, 3)
    inds = np.full([1, subj_locations.shape[0]], np.nan)
    for i in range(subj_locations.shape[0]):
        possible_locations = np.ones([all_locations.shape[0], 1])
        try:
            for c in range(all_locations.shape[1]):
                possible_locations[all_locations[:, c] != subj_locations[i, c], :] = 0
            inds[0, i] = np.where(possible_locations == 1)[0][0]
        except:
            pass
    inds = inds[~np.isnan(inds)]
    return [int(x) for x in inds]


def known_unknown(fullarray, knownarray, subarray=None, electrode=None):
    """
        This finds the indices for known and unknown electrodes in the full array of electrode locations

        Parameters
        ----------
        fullarray : ndarray
            Full array of electrode locations - All electrodes that pass the kurtosis test

        knownarray : ndarray
            Subset of known electrode locations  - Subject's electrode locations that pass the kurtosis test (in the leave one out case, this is also has the specified location missing)

        subarray : ndarray
            Subject's electrode locations (all)

        electrode : str
            Index of electrode in subarray to remove (in the leave one out case)

        Returns
        ----------
        known_inds : list
            List of known indices

        unknown_inds : list
            List of unknown indices

        """
    ## where known electrodes are located in full matrix
    known_inds = get_rows(np.round(fullarray, 3), np.round(knownarray, 3))
    ## where the rest of the electrodes are located
    unknown_inds = list(set(range(np.shape(fullarray)[0])) - set(known_inds))
    if not electrode is None:
        ## where the removed electrode is located in full matrix
        rm_full_ind = get_rows(np.round(fullarray, 3), np.round(subarray[int(electrode)], 3))
        ## where the removed electrode is located in the unknown index subset
        rm_unknown_ind = np.where(np.array(unknown_inds) == np.array(rm_full_ind))[0].tolist()
        return known_inds, unknown_inds, rm_unknown_ind
    else:
        return known_inds, unknown_inds



def _count_overlapping(X, Y):
    """
    Finds overlapping rows in two matrices

    Parameters
    ----------
    X : Numpy array of reference data

    Y : Numpy array of to-be-tested data

    Returns
    ----------
    results : ndarray
        Array of length Y.shape[0] with 0s and 1s, where 1s denote rows in Y that are also in X
    """

    return np.sum([(Y == x).all(1) for idx, x in X.iterrows()], 0).astype(bool)


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
    if locs.empty:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs)
    else:

        if label is not None:

            label = list(label)
            for i, v in enumerate(label):
                if v == 'observed':
                    label[i] = [0, 0, 1]
                elif v == 'removed':
                    label[i] = [0.0, 0.75, 0.75]
                else:
                    label[i] = [1, 0, 1]
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

    return Y, R, {'header': hdr, 'unscaled_timing':True}


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




def _plot_borderless(x, savefile=None, vmin=-1, vmax=1, width=1000, dpi=100, cmap='Spectral'):
    _close_all()
    width *= (1000.0 / 775.0)  # account for border
    height = (775.0 / 755.0) * float(width) * float(x.shape[0]) / float(x.shape[1])  # correct height/width distortion

    fig = plt.figure(figsize=(width / float(dpi), height / float(dpi)), dpi=dpi)

    if len(x.shape) == 2:
        plt.pcolormesh(x, vmin=float(vmin), vmax=float(vmax), cmap=cmap)
    else:
        plt.imshow(x)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.set_frameon(False)

    if not savefile == None:
        fig.savefig(savefile, figsize=(width / float(dpi), height / float(dpi)), bbox_inches='tight', pad_inches=0,
                    dpi=dpi)
    return fig


def _plot_big_matrix(X, outfile, max_blocksize=1000, width=1000, vmin=-1, vmax=1):
    if os.path.isfile(outfile):
        img = plt.imread(outfile)
        _plot_borderless(img)
        return img

    tmpdir1, fname = os.path.split(outfile)
    tmpdir2, tmp = os.path.splitext(fname)
    tmpdir = os.path.join(tmpdir1, tmpdir2)
    tmp_fname = os.path.join(tmpdir, 'tmp.png')
    if not os.path.isdir(tmpdir):
        delete_tmpdir = True
        os.makedirs(tmpdir)
    else:
        delete_tmpdir = False

    def carve_bign(n):
        starts = range(1, n, max_blocksize)
        ends = starts[1:]
        ends.append(n)
        ends = np.unique(ends)
        starts = np.array(starts) - 1
        return starts, ends

    row_starts, row_ends = carve_bign(X.shape[0])
    col_starts, col_ends = carve_bign(X.shape[1])
    n = np.max(X.shape)

    for row in np.arange(len(row_starts)):
        for col in np.arange(len(col_starts)):
            next_block = X[row_starts[row]:row_ends[row], col_starts[col]:col_ends[col]]
            next_width = float(width) * float(col_ends[col] - col_starts[col]) / float(X.shape[1])
            _plot_borderless(next_block, tmp_fname, vmin=vmin, vmax=vmax, width=next_width, dpi=10)

            next_img = plt.imread(tmp_fname)
            next_img = np.flipud(next_img)

            if col == 0:
                row_img = next_img
            else:
                row_img = _safe_cat(row_img, next_img, 1)
            if n > 1e4:
                print('.', end='')
        if row == 0:
            full_img = row_img
        else:
            full_img = _safe_cat(full_img, row_img, 0)
        if n > 1e4:
            print('', end='\n')

    if delete_tmpdir:
        shutil.rmtree(tmpdir)
    else:
        os.remove(tmp_fname)

    _plot_borderless(full_img, outfile);
    return full_img

def _safe_cat(a, b, axis):
    dims = list(set(np.arange(a.ndim)) - set([axis]))
    for d in dims:
        if a.shape[d] > b.shape[d]:
            b = _padder(b, a, d)
        elif a.shape[d] < b.shape[d]:
            a = _padder(a, b, d)
    return np.concatenate((a, b), axis=axis)


def _padder(a, b, dims):
    if not np.iterable(dims):
        dims = [dims]
    dims = np.array(dims)
    dims[dims < 0] = 0
    dims = dims.tolist()

    padding = np.array(map(lambda x: int(x in dims), np.arange(a.ndim))) * (np.array(b.shape) - np.array(a.shape))
    padding = padding * (padding > 0)
    return np.pad(a, zip(np.zeros([1, a.ndim], dtype=int).tolist()[0], padding.tolist()), 'mean')

def _close_all():
    figs = plt.get_fignums()
    for f in figs:
        plt.close(f)
