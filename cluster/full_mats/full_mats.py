
import superEEG as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template)

fig_dir = os.path.join(results_dir, 'figs')

try:
    os.stat(results_dir)
except:
    os.makedirs(results_dir)

try:
    os.stat(fig_dir)
except:
    os.makedirs(fig_dir)


# load locations for model

gray = se.load(intern(model_template))
gray_locs = gray.locs

file_name = os.path.basename(os.path.splitext(fname)[0])

if fname.split('.')[-1]=='bo':
    bo = se.load(fname)
    if se.filter_subj(bo):
        model = se.Model(bo, locs=gray_locs)
        model.save(fname=os.path.join(results_dir, file_name))
        model.plot()
        plt.savefig(os.path.join(fig_dir, file_name))
        print('done')

    else:
        print(file_name + '_filtered')
else:
    print('unknown file type')

