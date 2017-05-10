
class Brain(object):
    """
    Brain data object for the superEEG package

    Details about the Brain object.

    Parameters
    ----------

    data : Pandas DataFrame
        Samples x electrodes df containing the EEG data

    locs : Panadas DataFrame
        MNI coordinate (x,y,z) by electrode df containing electrode locations

    session : Pandas Series
        Samples x 1 array containing session identifiers

    sample_rate : float
        Sample rate of the data

    meta : dict
        Optional dict containing whatever you want

    Attributes
    ----------

    n_elecs : int
        Number of electrodes

    n_secs : float
        Amount of data in seconds

    n_sessions : int
        Number of sessions

    session_labels : list
        Label for each session

    Methods
    ----------

    ???

    Returns
    ----------

    brain : Brain data object
        Instance of Brain data object containing subject data

    """

    def __init__(self):
