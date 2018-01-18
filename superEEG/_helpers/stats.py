from __future__ import division
import multiprocessing
import copy
import numpy.matlib as mat
from scipy.stats import kurtosis, zscore, pearsonr
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import pandas as pd
from scipy import linalg
from joblib import Parallel, delayed
import nibabel as nb
import numpy as np
from nilearn.input_data import NiftiMasker

#from sklearn.neighbors import NearestNeighbors
#from sklearn.decomposition import PCA
#import seaborn as sns
#import tensorflow as tf


def apply_by_file_index(bo, xform, aggregator):

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
            results = xform(bo.get_data()[bo.sessions==session, :])
        else:
            results = aggregator(results, xform(bo.get_data()[bo.sessions==session, :]))

    return results

def kurt_vals(bo):
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

    return apply_by_file_index(bo, kurtosis, aggregate)

def get_corrmat(bo):
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
    def aggregate(p, n):
        return p+n

    def zcorr(x):
        return r2z(1 - squareform(pdist(x.T, 'correlation')))

    summed_zcorrs = apply_by_file_index(bo, zcorr, aggregate)

    return z2r(summed_zcorrs / len(bo.sessions.unique()))

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
    with np.errstate(invalid='ignore', divide='ignore'):
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
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


def rbf(x, center, width=20):
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

def expand_corrmat_fit(C, weights):
    """
    Gets full correlation matrix

    Parameters
    ----------
    C : Numpy array
        Subject's correlation matrix

    weights : Numpy array
        Weights matrix calculated using rbf function matrix

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

    vals = range(s, n)
    for x in vals:
        xweights = weights[x, :]

        vals = range(x)
        for y in vals:

            yweights = weights[y, :]

            next_weights = np.outer(xweights, yweights)
            next_weights = next_weights - np.triu(next_weights)

            W[x, y] = np.sum(next_weights)
            K[x, y] = np.sum(Z * next_weights)
    return (K + K.T), (W + W.T)

def expand_corrmat_predict(C, weights):
    """
    Gets full correlation matrix

    Parameters
    ----------
    C : Numpy array
        Subject's correlation matrix

    weights : Numpy array
        Weights matrix calculated using rbf function matrix

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
        delayed(compute_coord)(coord, weights, Z) for coord in sliced_up)

    W[map(lambda x: x[0], sliced_up), map(lambda x: x[1], sliced_up)] = map(lambda x: x[0], results)
    K[map(lambda x: x[0], sliced_up), map(lambda x: x[1], sliced_up)] = map(lambda x: x[1], results)

    return (K + K.T), (W + W.T)

def compute_coord(coord, weights, Z):

    next_weights = np.outer(weights[coord[0], :], weights[coord[1], :])
    next_weights = next_weights - np.triu(next_weights)

    return np.sum(next_weights), np.sum(Z * next_weights)


def reconstruct_activity(bo, K):
    """
    Reconstruct activity - need to add chunking option here
    """
    s = K.shape[0]-bo.locs.shape[0]
    Kba = K[:s,s:]
    Kaa = K[s:,s:]
    Y = zscore(bo.get_data())
    return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)


def round_it(locs, places):
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
    Filter bad electrodes
    """
    thresh_bool = bo.kurtosis > threshold
    nbo = copy.copy(bo)
    nbo.data = bo.data.loc[:, ~thresh_bool]
    nbo.locs = bo.locs.loc[~thresh_bool]
    nbo.n_elecs = bo.data.shape[1]
    return nbo


def filter_subj(bo, measure='kurtosis', threshold=10):
    """
    Filter subjects based on filter measure (use only if 2 or more electrodes pass thresholding)
    """
    if not bo.meta is None:
        thresh_bool = bo.kurtosis > threshold
        if sum(~thresh_bool)<2:
            print(bo.meta + ': not enough electrodes pass threshold')
        else:
            return bo.meta
    else:
        print('no meta data for brain object')


def corr_column(X, Y):
    return np.array([pearsonr(x, y)[0] for x, y in zip(X.T, Y.T)])


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


class BrainData:
    def __init__(self, fname, mask_strategy='background'):
        self.fname = fname
        if len(self.fname) == 0:
            self.Y = []
            self.R = []
            self.N = 0
            self.V = 0
            self.vox_size = (0, 0, 0)
            self.im_size = (0, 0, 0)
            self.mask = []
            self.img = []
        else:
            img = nb.load(self.fname)
            if not hasattr(img, 'dataobj'):
                print("Loading: " + self.fname + " [DISK READ]")
            else:
                print("Loading: " + self.fname + " [RAM CACHE]")

            self.mask = NiftiMasker(mask_strategy=mask_strategy)
            self.mask.fit(self.fname)

            hdr = img.get_header()
            S = img.get_sform()
            self.vox_size = hdr.get_zooms()
            self.im_size = img.shape

            if len(img.shape) > 3:
                self.N = img.shape[3]
            else:
                self.N = 1

            self.Y = self.mask.transform(self.fname)
            self.V = self.Y.shape[1]
            vmask = np.nonzero(np.array(
                np.reshape(self.mask.mask_img_.dataobj, (1, np.prod(self.mask.mask_img_.shape)), order='F')))[1]

            vox_coords = fullfact(img.shape[0:3])[vmask, :]
            self.matrix_coordinates = vox_coords

            self.R = np.array(vox_coords * S[0:3, 0:3] + np.tile(S[0:3, 3].T, (self.V, 1)))

def loadnii(fname, mask_strategy='background'):
    # if mask_strategy is 'background', treat uniformly valued voxels at the outer parts of the images as background
    # if mask_strategy is 'epi', use nilearn's background detection strategy: find the least dense point of the histogram, between fractions lower_cutoff and upper_cutoff of the total image histogram
    return BrainData(fname, mask_strategy)

def fullfact(dims):
    '''
    Replicates MATLAB's fullfact function (behaves the same way)
    '''
    vals = np.asmatrix(range(1, dims[0] + 1)).T
    if len(dims) == 1:
        return vals
    else:
        aftervals = np.asmatrix(fullfact(dims[1:]))
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
    from ..load import load
    from ..model import Model

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

# def compute_coord(coord, weights, Z):
#
#     xweights = weights[coord[0], :]
#     yweights = weights[coord[1], :]
#
#     next_weights = np.outer(xweights, yweights)
#     next_weights = next_weights - np.triu(next_weights)
#
#     w = np.sum(next_weights)
#
#     if w > 0:
#         return np.sum(Z * next_weights) / w
#     else:
#         return 0

# def compute_coord_tf(coord, weights, Z, sess):
#
#     xweights = weights[coord[0], :].reshape([weights.shape[1],1])
#     yweights = weights[coord[1], :].reshape([weights.shape[1],1])
#
#     next_weights = tf.matrix_band_part(tf.matmul(xweights, yweights.T), -1, 0)
#
#     w = tf.reduce_sum(next_weights)
#     k = tf.reduce_sum(tf.matmul(tf.constant(Z),next_weights))
#
#     result = k / w
#
#     return sess.run(result)
#
# def get_expanded_corrmat_tf(C, weights):
#     """
#     Gets full correlation matrix
#
#     Parameters
#     ----------
#     bo : Brain data object
#         Contains subject data, locs, other info
#
#     corrmat : len(n_elecs) x len(n_elecs) Numpy array
#         Subject's correlation matrix
#
#     weights : len()
#         Weights matrix calculated using rbf function - (len(R_subj)xlen(R_subj)) matrix
#
#     C_sub : ndarray
#         Subject level correlation matrix - (len(R_subj)xlen(R_subj)) matrix
#
#     """
#
#     # slice and dice
#     sliced_up = [(x, y) for x in range(weights.shape[0]) for y in range(x)]
#
#     Z = r2z(C)
#     Z[np.isnan(Z)] = 0
#
#     sess = tf.Session()
#
#     results = [compute_coord_tf(coord, weights, Z, sess) for coord in sliced_up]
#
#     return expand_matrix(results, weights)
# def reconstruct_activity_tf(bo, K):
#     """
#     Reconstruct activity using tensorflow
#     """
#     s = K.shape[0]-bo.locs.shape[0]
#     result = tf.matmul(tf.matmul(tf.constant(K[:s,s:], dtype='float32'),
#                         tf.matrix_inverse(tf.constant(K[s:,s:], dtype='float32'))),
#                         tf.constant(bo.get_data().T, dtype='float32'))
#     sess = tf.Session()
#     return sess.run(result).T
#### this version was to check how long one line would take
# def get_expanded_corrmat(C, weights, mode='fit'):
#     """
#     Gets full correlation matrix
#
#     Parameters
#     ----------
#     C : Numpy array
#         Subject's correlation matrix
#
#     weights : Numpy array
#         Weights matrix calculated using rbf function matrix
#
#     mode : str
#         Specifies whether to compute over all elecs (fit mode) or just new elecs
#         (predict mode)
#
#     Returns
#     ----------
#     numerator : Numpy array
#         Numerator for the expanded correlation matrix
#     denominator : Numpy array
#         Denominator for the expanded correlation matrix
#
#     """
#     C[np.eye(C.shape[0]) == 1] = 0
#     C[np.where(np.isnan(C))] = 0
#
#     n = weights.shape[0]
#     K = np.zeros([n, n])
#     W = np.zeros([n, n])
#     Z = C
#
#     if mode=='fit':
#         s = 0
#     elif mode=='predict':
#         s = C.shape[0]
#     else:
#         return []
#
#     vals_x = range(s, n)
#     for x in vals_x:
#         xweights = weights[x, :]
#
#         if mode=='fit':
#             vals_y = range(x)
#         elif mode == 'predict':
#             vals_y = range(1)
#         else:
#             return []
#         for y in vals_y:
#
#             yweights = weights[y, :]
#
#             next_weights = np.outer(xweights, yweights)
#             next_weights = next_weights - np.triu(next_weights)
#
#             W[x, y] = np.sum(next_weights)
#             K[x, y] = np.sum(Z * next_weights)
#
#     return (K + K.T), (W + W.T)

# def get_expanded_corrmat_parallel(C, weights, mode='fit'):
#     """
#     Gets full correlation matrix
#
#     Parameters
#     ----------
#     C : Numpy array
#         Subject's correlation matrix
#
#     weights : Numpy array
#         Weights matrix calculated using rbf function matrix
#
#     mode : str
#         Specifies whether to compute over all elecs (fit mode) or just new elecs
#         (predict mode)
#
#     Returns
#     ----------
#     numerator : Numpy array
#         Numerator for the expanded correlation matrix
#     denominator : Numpy array
#         Denominator for the expanded correlation matrix
#
#     """
#     C[np.eye(C.shape[0]) == 1] = 0
#     C[np.where(np.isnan(C))] = 0
#     n = weights.shape[0]
#     K = np.zeros([n, n])
#     W = np.zeros([n, n])
#     Z = C
#
#     if mode=='fit':
#         s = 0
#     elif mode=='predict':
#         s = C.shape[0]
#     else:
#         return []
#
#     ### to debug multiprocessing:
#
#     if mode =='predict':
#         sliced_up = [(x, y) for x in range(s, n) for y in range(x)]
#     else:
#         return []
#
#     results = Parallel(n_jobs=multiprocessing.cpu_count())(
#         delayed(compute_coord)(coord, weights, Z) for coord in sliced_up)
#
#     w, k = zip(*results)
#
#     for i, x in enumerate(sliced_up):
#         W[x[0], x[1]] = w[i]
#         K[x[0], x[1]] = k[i]
#
#
#     return (K + K.T), (W + W.T)

# def get_expanded_corrmat(C, weights):
#     """
#     Gets full correlation matrix
#
#     Parameters
#     ----------
#     bo : Brain data object
#         Contains subject data, locs, other info
#
#     corrmat : len(n_elecs) x len(n_elecs) Numpy array
#         Subject's correlation matrix
#
#     weights : len()
#         Weights matrix calculated using rbf function - (len(R_subj)xlen(R_subj)) matrix
#
#     C_sub : ndarray
#         Subject level correlation matrix - (len(R_subj)xlen(R_subj)) matrix
#
#     """
#
#     # slice and dice
#     sliced_up = [(x, y) for x in range(weights.shape[0]) for y in range(x)]
#
#     Z = r2z(C)
#     Z[np.isnan(Z)] = 0
#
#     results = Parallel(n_jobs=multiprocessing.cpu_count())(
#         delayed(compute_coord)(coord, weights, Z) for coord in sliced_up)
#
#     return expand_matrix(results, weights)

# def get_expanded_corrmat_lucy(C, weights, mode='fit'):
#     C[np.eye(C.shape[0]) == 1] = 0
#     C[np.where(np.isnan(C))] = 0
#
#     n = weights.shape[0]
#     s = C.shape[0]
#     K = np.zeros([n, n])
#     W = np.zeros([n, n])
#     Z = C
#
#     predict_mode = (mode == 'predict')
#
#     for x in range(n):
#         xweights = weights[x, :]
#         if predict_mode:
#             vals = range(x, n)
#         else:
#             vals = range(x)
#         for y in vals:
#             if predict_mode and (y < s): #this may be off by one index
#                 continue
#             yweights = weights[y, :]
#
#             next_weights = np.outer(xweights, yweights)
#             next_weights = next_weights - np.triu(next_weights)
#
#             W[x, y] = np.sum(next_weights)
#             K[x, y] = np.sum(Z * next_weights)
#
#     return (K + K.T), (W + W.T)

#
# def get_expanded_corrmat(C, weights, mode='fit'):
#     """
#     Gets full correlation matrix
#
#     Parameters
#     ----------
#     C : Numpy array
#         Subject's correlation matrix
#
#     weights : Numpy array
#         Weights matrix calculated using rbf function matrix
#
#     mode : str
#         Specifies whether to compute over all elecs (fit mode) or just new elecs
#         (predict mode)
#
#     Returns
#     ----------
#     numerator : Numpy array
#         Numerator for the expanded correlation matrix
#     denominator : Numpy array
#         Denominator for the expanded correlation matrix
#
#     """
#     C[np.eye(C.shape[0]) == 1] = 0
#     C[np.where(np.isnan(C))] = 0
#
#     n = weights.shape[0]
#     K = np.zeros([n, n])
#     W = np.zeros([n, n])
#     Z = C
#
#     if mode=='fit':
#         s = 0
#
#         vals = range(s, n)
#         for x in vals:
#             xweights = weights[x, :]
#
#             vals = range(x)
#             for y in vals:
#
#                 yweights = weights[y, :]
#
#                 next_weights = np.outer(xweights, yweights)
#                 next_weights = next_weights - np.triu(next_weights)
#
#                 W[x, y] = np.sum(next_weights)
#                 K[x, y] = np.sum(Z * next_weights)
#         return (K + K.T), (W + W.T)
#
#     elif mode=='predict':
#         s = C.shape[0]
#         sliced_up = [(x, y) for x in range(s, n) for y in range(x)]
#
#         results = Parallel(n_jobs=multiprocessing.cpu_count())(
#             delayed(compute_coord)(coord, weights, Z) for coord in sliced_up)
#
#         W[map(lambda x: x[0], sliced_up), map(lambda x: x[1], sliced_up)] = map(lambda x: x[0], results)
#         K[map(lambda x: x[0], sliced_up), map(lambda x: x[1], sliced_up)] = map(lambda x: x[1], results)
#
#         return (K + K.T), (W + W.T)
#
#     else:
#         return 'error: unknown mode entered for get_expand_corrmat'
# def expand_matrix(output_list, R_full):
#     """
#     This function expands output from the pooled RBF-weighted averages at each coordinate (index in the full matrix)
#
#     Parameters
#     ----------
#     output_list : list
#         results from pooled expand_corrmat
#
#     R_full : ndarray
#         Full list of coordinates that pass kurtosis threshold
#
#     Returns
#     ----------
#     results : ndarray
#         Expanded full matrix (len(R_full) x len(R_full))
#
#     """
#     ### convert output list to array
#     output_array = np.array(output_list)
#     ### initialize a full matrix (len(R_full) x len(R_full))
#     C_full = np.zeros([R_full.shape[0], R_full.shape[0]])
#     ### find indices of for the top triangle in the full matrix - use those indices to fill in the correponding values from output array to full matrix
#     C_full[np.tril_indices(R_full.shape[0], -1)] = output_array
#     ### expand to full matrix
#     return C_full + C_full.T + np.eye(C_full.shape[0])
##### both recons below were just used for debugging purposes
#
# def recon_no_expand(bo_sub, mo):
#     """
#     """
#     model = z2r(np.divide(mo.numerator, mo.denominator))
#     model[np.eye(model.shape[0]) == 1] = 1
#     known_locs = bo_sub.locs
#     known_inds = bo_sub.locs.index.values
#     unknown_locs = mo.locs.drop(known_inds)
#     unknown_inds = unknown_locs.index.values
#     Kba = model[unknown_inds, :][:, known_inds]
#     Kaa = model[:,known_inds][known_inds,:]
#     Y = zscore(bo_sub.get_data())
#     return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)
#
# def recon(bo_sub, mo):
#     """
#     """
#     mo[np.eye(mo.shape[0]) == 1] = 1
#     known_inds = bo_sub.locs.index.values
#     locs_inds = range(mo.shape[0])
#     unknown_inds = np.sort(list(set(locs_inds) - set(known_inds)))
#     Kba = mo[unknown_inds, :][:, known_inds]
#     Kaa = mo[:,known_inds][known_inds,:]
#     Y = zscore(bo_sub.get_data())
#     return np.squeeze(np.dot(np.dot(Kba, np.linalg.pinv(Kaa)), Y.T).T)