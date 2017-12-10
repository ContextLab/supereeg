
import superEEG as se
import numpy as np
#from superEEG._helpers.bookkeeping import sort_unique_locs
import glob
import sys
import os
from config import config
import pandas as pd


try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])

def sort_unique_locs(locs):
    if isinstance(locs, pd.DataFrame):
        unique_full_locs = np.vstack(set(map(tuple, locs.as_matrix())))
    elif isinstance(locs, np.ndarray):
        unique_full_locs = np.vstack(set(map(tuple, locs)))
    else:
        print('unknown location type')

    return unique_full_locs[unique_full_locs[:, 0].argsort(),]

results_dir = config['resultsdir']


data_dir = config['datadir']

bo_files = glob.glob(os.path.join(data_dir,'*.bo'))



union_locs = pd.DataFrame()
for b in bo_files:
    #bo = se.load(b)
    bo = se.filter_elecs(se.load(b))
    union_locs = union_locs.append(bo.locs, ignore_index=True)


locations = sort_unique_locs(union_locs)

filepath=os.path.join(results_dir, 'union_locs.npy')

np.save(filepath, locations)


print('done')

