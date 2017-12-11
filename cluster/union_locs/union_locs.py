
import superEEG as se
import numpy as np
#from superEEG._helpers.bookkeeping import sort_unique_locs
import glob
import sys
import os
from config import config
import pandas as pd

## this script takes iterates over brain objects, filters them based on kurtosis,
## then compiles the clean electrodes into a numpy array as well as a list of the contributing brain objects

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
model_data = []
for b in bo_files:
    model_data.append(se.filter_subj(se.load(b)))
    print(b)

print(model_data)

for bo in model_data:
    if bo == None:
        continue
    else:
        ## for only the union electrode locations that pass kurtosis threshold:
        bo = se.filter_elecs(se.load(os.path.join(data_dir, bo + '.bo')))
        ## for all locations
        #bo = se.load(b)


    union_locs = union_locs.append(bo.locs, ignore_index=True)


locations = sort_unique_locs(union_locs)

filepath=os.path.join(results_dir, 'union_locs.npz')

np.savez(filepath, locs = locations, subjs = model_data)

print('done')

