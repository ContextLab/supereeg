from __future__ import division
import warnings
import scipy
import numpy as np
import pandas as pd
from sklearn import datasets
from .brain import Brain

def simulate_locations(n_elecs=10, set_random_seed=False):
    """
    Simulate iEEG locations

    Parameters
    ----------
    n_elecs : int
        Number of electrodes

    set_random_seed : bool or int
        Default False.  If True, set random seed to 123.  If int, set random
        seed to value.

    Returns
    ----------
    elecs : pd.DataFrame
        A location by coordinate (x,y,z) matrix of simulated electrode locations

    """
    if set_random_seed:
        if isinstance(set_random_seed, int):
            np.random.seed(set_random_seed)
        else:
            np.random.seed(123)


    sim_locs = np.array([[np.random.randint(-50, 50), np.random.randint(-50, 50),
               np.random.randint(-50, 50)] for i in range(n_elecs)])

    locs = np.unique(sim_locs, axis=0)

    return pd.DataFrame(locs, columns=['x', 'y', 'z'])

def simulate_model_bos(n_samples=1000, locs=None, sample_locs=None, cov='random',
                       sample_rate=1000, sessions=None, meta=None, noise=.1,
                       set_random_seed=False):
    """
    Simulate brain object

    Parameters
    ----------
    n_samples : int
        Number of time samples

    locs :  np.ndarray or pd.DataFrame
         A location by coordinate (x,y,z) matrix of simulated electrode locations

    sample_locs : int
        Number of subsampled electrode location to create each brain object

    cov : str or np.array

        The covariance structure of the data.

            If 'eye', the covariance will be the identity matrix.

            If 'toeplitz', the covariance will be a toeplitz matrix.

            If 'random', uses a random semidefinite matrix with a set random seed.

            If 'distance'calculates the euclidean distance between each electrode.

        You can also pass a custom covariance matrix by simply passing
        numpy array that is n_elecs by n_elecs

    sample_rate : int or float
        Sample rate (Hz)

    sessions : list
        Sesssions

    meta : str
        Meta info

    noise : int or float
        Noise added to simulation

    set_random_seed : bool or int
        Default False.  If True, set random seed to 123.  If int, set random seed to value.

    Returns
    ----------
    bo : Brain data object
        Instance of Brain data object containing simulated subject data and locations

    """

    data, sub_locs= simulate_model_data(n_samples=n_samples, locs=locs, sample_locs=sample_locs, cov=cov, noise=noise, set_random_seed=set_random_seed)

    return Brain(data=data, locs=sub_locs, sample_rate=sample_rate,
                 sessions=sessions, meta=meta)


def simulate_model_data(n_samples=1000, n_elecs=170, locs=None, sample_locs=None,
                        cov='random', noise=.1, set_random_seed=False):
    """
    Simulate iEEG data

    Parameters
    ----------
    n_samples : int
        Number of time samples

    n_elecs : int
        Number of electrodes

    locs :  np.ndarray or pd.DataFrame
         A location by coordinate (x,y,z) matrix of simulated electrode locations

    sample_locs : int
        Number of subsampled electrode location to create each brain object

    cov : str or np.array

        The covariance structure of the data.

            If 'eye', the covariance will be the identity matrix.

            If 'toeplitz', the covariance will be a toeplitz matrix.

            If 'random', uses a random semidefinite matrix with a set random seed.

            If 'distance'calculates the euclidean distance between each electrode.

        You can also pass a custom covariance matrix by simply passing
        numpy array that is n_elecs by n_elecs

    noise : int or float
        Noise added to simulation

    set_random_seed : bool or int
        Default False (choose a random seed).  If True, set random seed to 123.  If int, set random seed to the specified value.

    Returns
    ----------
    data : np.ndarray
        A samples by number of electrodes array of simulated iEEG data

    sub_locs : pd.DataFrame
        A location by coordinate (x,y,z) matrix of simulated electrode locations

    """
    if set_random_seed:
        if isinstance(set_random_seed, int):
            np.random.seed(set_random_seed)
        else:
            np.random.seed(123)
            set_random_seed = 123
    else:
        set_random_seed = None

    if type(locs) is np.ndarray:
        locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
    if locs is not None and cov is 'distance':
        R = 1 - scipy.spatial.distance.cdist(locs, locs, metric='euclidean')
        R -= np.min(R)
        R /= np.max(R)
        cov = 2*R - 1
    if sample_locs is not None:
        R = create_cov(cov, n_elecs=len(locs))
        n = np.random.normal(0, noise, len(locs))
        R = R+n*n.T
        sub_locs = locs.sample(sample_locs, random_state=set_random_seed).sort_values(['x', 'y', 'z'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_data = np.random.multivariate_normal(np.zeros(len(locs)), R, size=n_samples)
        data = full_data[:, sub_locs.index]
        return data, sub_locs
    else:
        R = create_cov(cov, n_elecs=len(locs))
        n_elecs = len(locs)

        return np.random.multivariate_normal(np.zeros(n_elecs), R, size=n_samples), locs

def simulate_bo(n_samples=1000, n_elecs=10, locs=None, cov='random',
                sample_rate=1000, sessions=None, meta=None, noise=.1,
                random_seed=False):
    """
    Simulate brain object

    Parameters
    ----------
    n_samples : int
        Number of time samples

    n_elecs : int
        Number of electrodes

    locs :  np.ndarray or pd.DataFrame
         A location by coordinate (x,y,z) matrix of simulated electrode locations

    cov : str or np.array

        The covariance structure of the data.

            If 'eye', the covariance will be the identity matrix.

            If 'toeplitz', the covariance will be a toeplitz matrix.

            If 'random', uses a random semidefinite matrix with a set random seed.

            If 'distance'calculates the euclidean distance between each electrode.

        You can also pass a custom covariance matrix by simply passing
        numpy array that is n_elecs by n_elecs.

    sample_rate : int or float
        Sample rate (Hz)

    sessions : list
        Sesssions

    meta : str
        Meta info

    noise : int or float
        Noise added to simulation

    random_seed : bool or int
        Default False.  If True, set random seed to 123.  If int, set random
        seed to value.

    Returns
    ----------
    bo : Brain data object
        Instance of Brain data object containing simulated subject data and
        locations

    """
    if locs is None:
        locs =  simulate_locations(n_elecs=n_elecs)
    else:
        n_elecs=locs.shape[0]

    if type(sessions) is int:
        sessions = np.sort([x + 1 for x in range(sessions)] * int(np.floor(np.divide(n_samples, sessions))))

    data, locs = simulate_model_data(n_samples=n_samples, n_elecs=n_elecs, locs=locs, cov=cov, noise=noise, set_random_seed=random_seed)

    return Brain(data=data, locs=locs, sample_rate=sample_rate,
                 sessions=sessions, meta=meta)

def create_cov(cov, n_elecs=10):
    """
    Creates covariance matrix of specified type

    Parameters
    ----------
    cov : str or np.array
        The covariance structure of the data.

            If 'eye', the covariance will be the identity matrix.

            If 'toeplitz', the covariance will be a toeplitz matrix.

            If 'random', uses a random semidefinite matrix with a set random seed.

            If 'distance'calculates the euclidean distance between each electrode.

        You can also pass a custom covariance matrix by simply passing
        numpy array that is n_elecs by n_elecs.

    n_elecs : int
        Number of electrodes

    Returns
    ----------
    R : np.ndarray
        Numpy array containing the covariance structure in the specified
        dimension (n_elecs x n_elecs)

    """
    if cov is 'eye':
        R = np.eye(n_elecs)
    elif cov is 'toeplitz':
        R = scipy.linalg.toeplitz(np.linspace(0, 1, n_elecs)[::-1])
    elif cov is 'random':
        R = datasets.make_spd_matrix(n_elecs, random_state=1)
        R -= np.min(R)
        R /= np.max(R)
    elif isinstance(cov, np.ndarray):
        R = cov
    return R
