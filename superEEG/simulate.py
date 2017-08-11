import scipy
import numpy as np

def simulate_data(n_samples=1000, n_elecs=10, cov='eye'):
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

    if cov is 'eye':
        R = np.eye(n_elecs)
    elif cov is 'toeplitz':
        R = scipy.linalg.toeplitz(np.linspace(0, 1, n_elecs)[::-1])
    elif isinstance(cov, np.ndarray):
        R = cov

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
