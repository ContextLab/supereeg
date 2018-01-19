
import superEEG as se
import numpy as np
import glob
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template)

fig_dir = os.path.join(results_dir, 'figs')

try:
    if not os.path.exists(os.path.dirname(results_dir)):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

try:
    if not os.path.exists(os.path.dirname(fig_dir)):
        os.makedirs(fig_dir)
except OSError as err:
   print(err)

# load locations for model
if model_template == 'pyFR_union':
    data = np.load(os.path.join(config['pyFRlocsdir'],'pyFR_k10_locs.npz'))
    locs = data['locs']
    gray_locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
else:
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

