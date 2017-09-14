import scipy
import numpy as np
from .brain import Brain
from .model import Model

def simulate_data(n_samples=1000, n_elecs=10, locs=None, cov='distance'):
    """
    Simulate iEEG data

    Parameters
    ----------

    n_samples : int
        Number of time samples

    n_elecs : int
        Number of electrodes

    cov : str or np.array
        The covariance structure of the data.  if 'eye', the covariance will be
        the identity matrix.  If 'toeplitz', the covariance will be a toeplitz
        matrix.  You can also pass a custom covariance matrix by simply passing
        numpy array that is n_elecs by n_elecs

    Returns
    ----------

    data: np.ndarray
        A samples by number of electrods array of simulated iEEG data

    """
    if locs is not None:
        R = 1 - scipy.spatial.distance.cdist(locs, locs, metric='euclidean')
        R -= np.min(R)
        R /= np.max(R)
        R *= 2*R - 1
    else:
        R = create_cov(cov, n_elecs=n_elecs)

    return np.random.multivariate_normal(np.zeros(n_elecs), R, size=n_samples)

def simulate_locations(n_elecs=10):
    """
    Simulate iEEG locations

    Parameters
    ----------

    n_elecs : int
        Number of electrodes

    Returns
    ----------

    elecs : np.ndarray
        A location by coordinate (x,y,z) matrix of simulated electrode locations
    """

    return np.array([[np.random.randint(-80, 80), np.random.randint(-80, 80),
               np.random.randint(-80, 80)] for i in range(n_elecs)])

def simulate_bo(n_samples=1000, n_elecs=10, locs=None, cov='distance',
                sample_rate=1000, sessions=None, meta=None):
    """
    Simulate brain object

    Parameters
    ----------

    n_elecs : int
        Number of electrodes

    Returns
    ----------

    elecs : np.ndarray
        A location by coordinate (x,y,z) matrix of simulated electrode locations
    """
    if not locs:
        locs =  simulate_locations(n_elecs=n_elecs)
    else:
        n_elecs=locs.shape[0]
        
    data = simulate_data(n_samples=n_samples, n_elecs=n_elecs, locs=locs, cov=cov)

    return Brain(data=data, locs=locs, sample_rate=sample_rate,
                 sessions=sessions, meta=meta)

def simulate_model(n_subs=10, n_samples=1000, n_elecs=10, cov='eye'):
    """
    Simulate a model object

    Parameters
    ----------

    n_subs : int
        Number of subjects

    n_subs : int
        Number of subjects

    Returns
    ----------

    elecs : np.ndarray
        A location by coordinate (x,y,z) matrix of simulated electrode locations
    """

    # create covariance matrix
    cov = create_cov(cov, n_elecs=n_elecs)

    #
    bos = [simulate_bo(n_samples=n_samples, n_elecs=n_elecs, cov=cov) for i in range(n_subs)]

    return Model(data=bos)


def create_cov(cov, n_elecs=10):
    """
    Creates covariance matrix of specified type
    """
    if cov is 'eye':
        R = np.eye(n_elecs)
    elif cov is 'toeplitz':
        R = scipy.linalg.toeplitz(np.linspace(0, 1, n_elecs)[::-1])
    elif isinstance(cov, np.ndarray):
        R = cov

    return R
