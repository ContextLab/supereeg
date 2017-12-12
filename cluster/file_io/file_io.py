
import superEEG as se
import numpy as np
from superEEG._helpers.stats import tal2mni
import glob
import sys
import os
from config import config

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])

def npz2bo(infile):

    with open(infile, 'rb') as handle:
        f = np.load(handle)
        f_name = os.path.splitext(os.path.basename(infile))[0]
        data = f['Y']
        sample_rate = f['samplerate']
        sessions = f['fname_labels']
        locs = tal2mni(f['R'])
        meta = f_name

    return se.Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta=meta)

results_dir = config['resultsdir']


fname = sys.argv[1]

file_name = os.path.basename(os.path.splitext(fname)[0])
bo = se.npz2bo(fname)

bo.save(filepath=os.path.join(results_dir, file_name))


print('done')
