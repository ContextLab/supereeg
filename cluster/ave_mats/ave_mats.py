
import superEEG as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config


model_template = sys.argv[1]

model_dir = os.path.join(config['datadir'], model_template)

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

model_data = glob.glob(os.path.join(model_dir,'*.mo'))


ave_model = se.model_compile(model_data)

ave_model.save(filepath=os.path.join(results_dir, model_template))

ave_model.plot()

plt.savefig(os.path.join(fig_dir, model_template))


print(ave_model.n_subs)

