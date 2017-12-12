
import superEEG as se
import numpy as np
import glob
import sys
import os
from config import config

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])


results_dir = config['resultsdir']


fname = sys.argv[1]

file_name = os.path.basename(os.path.splitext(fname)[0])
bo = se.npz2bo(fname)

bo.save(filepath=os.path.join(results_dir, file_name))


print('done')
