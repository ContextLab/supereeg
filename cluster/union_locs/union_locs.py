
import superEEG as se
import numpy as np
from superEEG._helpers.bookkeeping import sort_unique_locs
import glob
import sys
import os
from config import config
import pandas as pd


try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])


results_dir = config['resultsdir']


data_dir = config['datadir']

bo_files = glob.glob(os.path.join(data_dir,'*.bo'))

union_locs = pd.DataFrame()
for b in bo_files:
    bo = se.load(b)
    union_locs = union_locs.append(bo.locs, ignore_index=True)


locations = sort_unique_locs(union_locs)

filepath=os.path.join(results_dir, 'union_locs.npy')

np.save(filepath, locations)


print('done')

