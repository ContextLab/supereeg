#!/usr/bin/env python

import numpy as np
from glob import glob as lsdir
from stats import tal2mni, uniquerows, round_it
import os
from scipy.spatial.distance import squareform


def get_rows(all_locations, subj_locations):
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


# TODO: this has not been debugged
# datadir can either be a file OR a directory
def get_locs(datadir):
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
    return (myarray == myrow).all(-1).any()


def remove_electrode(subkarray, subarray, electrode):
    rm_ind = get_rows(subkarray, subarray[int(electrode)])
    return np.delete(subkarray, rm_ind, 0)


def known_unknown(fullarray, knownarray, subarray=None, electrode=None):
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


## Subj_matrix needs to be squareformed and id matrix added back in
def alter_avemat(Average_matrix, Subj_matrix):
    summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
    count_removed = Average_matrix['n'] - 1
    C_est = squareform(Subj_matrix['C_est'], checks=False)
    C_est[np.where(np.isnan(C_est))] = 0
    return (summed_matrix - (C_est + np.eye(C_est.shape[0]))) / count_removed

def get_parent_dir(directory):
    import os
    return os.path.dirname(directory)

def get_grand_parent_dir(directory):
    import os
    return os.path.dirname(os.path.dirname(directory))

def flatten_arrays(array_of_arrays):
    flattened_array = []
    for sublist in array_of_arrays:
        for item in sublist:
            flattened_array.append(item)
    return flattened_array
