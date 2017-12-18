import superEEG as se
import numpy as np
from superEEG._helpers.bookkeeping import sort_unique_locs
import glob
import os
from config import config
import pandas as pd
from nilearn import plotting as ni_plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')

## this script iterates over brain objects, filters them based on kurtosis value,
## then compiles the clean electrodes into a numpy array as well as a list of the contributing brain objects

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])


results_dir = config['resultsdir']

data_dir = config['datadir']

bo_files = glob.glob(os.path.join(data_dir,'*.bo'))

model_data = []
for b in bo_files:
    model_data.append(se.filter_subj(se.load(b)))
    print(b)

print(model_data)


union_locs = pd.DataFrame()
for bo in model_data:
    if bo == None:
        continue
    else:
    ## for only the union electrode locations that pass kurtosis threshold:
        bo = se.filter_elecs(se.load(os.path.join(data_dir, bo + '.bo')))

    union_locs = union_locs.append(bo.locs, ignore_index=True)


locations = sort_unique_locs(union_locs)

filepath=os.path.join(results_dir, 'pyFR_k10_locs.npz')

np.savez(filepath, locs = locations, subjs = model_data)

pdfpath=os.path.join(results_dir, 'pyFR_k10_locs.pdf')

ni_plt.plot_connectome(np.eye(locations.shape[0]), locations, display_mode='lyrz', output_file=pdfpath, node_kwargs={'alpha':0.5, 'edgecolors':None}, node_size=10, node_color = np.ones(locations.shape[0]))

print('done')
