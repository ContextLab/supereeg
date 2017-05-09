#!/usr/bin/env python

import numpy as np
from glob import glob as lsdir
from stats import tal2mni, uniquerows, round_it
import os
from scipy.spatial.distance import squareform


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
    return map(int, inds)



# datadir can either be a file OR a directory
# I dont think this is in use
def get_locs(datadir):
    """
        This function compiles electrode locations

        Parameters
        ----------
        datadir : file or directory
            location or file containing electrode locations

        Returns
        ----------
        results : ndarray
            Compiled electrode locations

        """
    if datadir[-3:] == 'npz':
        files = (datadir)
    else:
        files = lsdir(os.path.join(datadir, '*.npz'))

    R = []
    for f in files:
        data = np.load(f, mmap_mode='r')
        if len(R) == 0:
            R = data['R']
        else:
            R = np.vstack((R, data['R']))

    R = tal2mni(uniquerows(R))
    return R[R[:, 0].argsort(),]


def row_in_array(myarray, myrow):
    """
        Looks to see if a row (electrode location) is in the bigger array

        Parameters
        ----------
        myarray : ndarray
            Larger array of electrode locations

        myrow : ndarray
            Specific row to find

        Returns
        ----------
        results : bool
            True if row in array; False if not

        """
    return (myarray == myrow).all(-1).any()


def remove_electrode(subkarray, subarray, electrode):
    """
        Removes electrode from larger array

        Parameters
        ----------
        subkarray : ndarray
            Subject's electrode locations that pass the kurtosis test

        subarray : ndarray
            Subject's electrode locations (all)

        electrode : str
            Index of electrode in subarray to remove

        Returns
        ----------
        results : ndarray
            Subject's electrode locations that pass kurtosis test with electrode removed

        """
    rm_ind = get_rows(subkarray, subarray[int(electrode)])
    return np.delete(subkarray, rm_ind, 0)


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
    known_inds = get_rows(round_it(fullarray, 3), round_it(knownarray, 3))
    ## where the rest of the electrodes are located
    unknown_inds = list(set(range(np.shape(fullarray)[0])) - set(known_inds))
    if not electrode is None:
        ## where the removed electrode is located in full matrix
        rm_full_ind = get_rows(round_it(fullarray, 3), round_it(subarray[int(electrode)], 3))
        ## where the removed electrode is located in the unknown index subset
        rm_unknown_ind = np.where(np.array(unknown_inds) == np.array(rm_full_ind))[0].tolist()
        return known_inds, unknown_inds, rm_unknown_ind
    else:
        return known_inds, unknown_inds



def alter_avemat(Average_matrix, Subj_matrix):
    """
        Removes one subject's full correlation matrix from the average correlation matrix

        Parameters
        ----------
        Average_matrix : npz file
            npz file contains the fields:
                average_matrix : the average full correlation matrix for all subjects (n)
                n : number of full correlation matrices that contributed to average matrix

        Subj_matrix : list
            Subject's squareformed full correlation matrix

        Returns
        ----------
        results : ndarray
            Average matrix with one subject's data removed

        """
    summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
    count_removed = Average_matrix['n'] - 1
    C_est = squareform(Subj_matrix['C_est'], checks=False)
    C_est[np.where(np.isnan(C_est))] = 0
    return (summed_matrix - (C_est + np.eye(C_est.shape[0]))) / count_removed

def get_parent_dir(directory):
    """
        Gives path to one directory up

        Parameters
        ----------
        directory : str
            Path to files

        Returns
        ----------
        results : str
            Path for one directory up

        """
    import os
    return os.path.dirname(directory)

def get_grand_parent_dir(directory):
    """
        Gives path to two directories up

        Parameters
        ----------
        directory : str
            Path to files

        Returns
        ----------
        results : str
            Path for two directories up

        """
    import os
    return os.path.dirname(os.path.dirname(directory))

