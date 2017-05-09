import glob
import os
import sys
import scipy.io
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy import linalg
import numpy as np
from plot import brain_net_viewer_format_color
from bookkeeping import get_parent_dir, get_grand_parent_dir
import matplotlib as mpl
# mpl.use('Agg')
import pylab as plt
import pandas as pd
import re
import seaborn as sb
# import hypertools as hyp

fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs/paper')

def tal2mni(r):
    # convert a series of Talairach coordinates (rows of r) into MNI coordinates
    rotmat = np.array([[1, 0, 0, 0], [0, 0.9988, 0.0500, 0], [0, -0.0500, 0.9988, 0], [0, 0, 0, 1.0000]])
    up = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.9200, 0], [0, 0, 0, 1.0000]])
    down = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.8400, 0], [0, 0, 0, 1.0000]])

    # return np.dot(down, np.dot(up, np.dot(rotmat, np.pad(r, (0, 1), 'constant', constant_values=(0, 1)).T)))
    return np.dot(down, np.dot(up, np.dot(rotmat, np.c_[r, np.ones(r.shape[0], dtype=np.int)].T)))


def new_attempt(r):
    rotmat = np.array([[1, 0, 0, 0], [0, 0.9988, 0.0500, 0], [0, -0.0500, 0.9988, 0], [0, 0, 0, 1.0000]])
    up = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.9200, 0], [0, 0, 0, 1.0000]])
    down = np.array([[0.9900, 0, 0, 0], [0, 0.9700, 0, 0], [0, 0, 0.8400, 0], [0, 0, 0, 1.0000]])

    inpoints = np.c_[r, np.ones(r.shape[0], dtype=np.float)].T
    tmp = inpoints[2,:] < 0
    inpoints[:,tmp] = linalg.solve(np.dot(rotmat, down), inpoints[:, tmp])
    inpoints[:,~tmp] = linalg.solve(np.dot(rotmat, up), inpoints[:, ~tmp])

    return inpoints[0:3, :].T

def euc_dist(a, b):
    diffs = (a-b)**2
    distances = diffs.sum(axis=1)
    distances = np.sqrt(distances)
    return distances

tal_locations = scipy.io.loadmat(os.path.join(get_parent_dir(os.getcwd()), 'all_locations.mat'))
tal_locs = tal_locations['R']
mni_locations = scipy.io.loadmat(os.path.join(get_parent_dir(os.getcwd()), 'mni_locations.mat'))
mni_locs = mni_locations['R']
TAL10 = tal_locs[0:-1:10, :]
MNI10 = mni_locs[0:-1:10, :]
brain_net_tal_locs = brain_net_viewer_format_color(TAL10,1)
brain_net_mni_locs = brain_net_viewer_format_color(MNI10,1)
np.savetxt(os.path.join(fig_dir, 'brain_net_locs_TAL.node'), brain_net_tal_locs)
np.savetxt(os.path.join(fig_dir, 'brain_net_locs_MNI.node'), brain_net_mni_locs)

test_tal = tal_locs[0:10]
test_mni = mni_locs[0:10]
calc_mni = new_attempt(test_tal)
diff_dist = euc_dist(test_mni, calc_mni)

full_mni = new_attempt(tal_locs)
full_diff = euc_dist(mni_locs, full_mni)
plt.plot(full_diff)

FIXED10 = full_mni[0:-1:10, :]
brain_net_fixed_mni_locs = brain_net_viewer_format_color(FIXED10,1)
np.savetxt(os.path.join(fig_dir, 'brain_net_locs_fixed.node'), brain_net_fixed_mni_locs)