
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

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)
locs = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10)
bo = se.Brain(data=data, locs=locs)

# load locations for model
### this weird work around is necessary because there's an issue using a variable for a string in an argument

# if model_template == 'mini_model':
#     gray = se.load('mini_model')
#     gray_locs = gray.locs
#
# elif model_template == 'pyFR_locs':
#     data = np.load(os.path.join(config['startdir'],'pyFR_locs/results/pyFR_k10_locs.npz'))
#     #gray_locs = se.load('pyFR_union')
#
#
# elif model_template == 'gray_mask_6mm_brain':
#     gray = se.load('gray_mask_6mm_brain')
#     gray_locs = gray.locs
# else:
#     gray = se.load('mini_model')
#     gray_locs = gray.locs

### think I resolved the issue, but just to be sure:

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

# work around if not brain objects, but :
# if fname.split('.')[-1]=='bo':
#     bo = se.filter_elecs(se.load(fname))
#
# else:
#     print('unknown file type')
#
# if bo.locs.shape[0] > 1:
#
#     model = se.Model(bo, locs = gray_locs)
#
#
#     print('done')
#
# else:
#     print(file_name + 'filtered')

# model_data = []
# bo_files = glob.glob(os.path.join('/Users/lucyowen/Desktop/analysis/bo','*.bo'))
# # bo_files = glob.glob(os.path.join('/idata/cdl/data/ECoG/pyFR/data/bo','*.bo'))
# # for i, b in enumerate(bo_files):
# #     if i < 2:
# #         bo = se.load(b)
# #         model_data.append(se.Brain(data=bo.data, locs=bo.locs))
# #     elif i == 2:
# #         bo = se.load(b)
# #         model_data.append(se.Brain(data=bo.data, locs=bo.locs))
# #         model = se.Model(data=model_data)
# #     else:
# #         bo = se.load(b)
# #         model = model.update(bo)
#
# model = se.Model([se.load(b) for b in bo_files[:2]])
# for b in bo_files[2:]:
#     model = model.update(se.load(b))
#
# #model = se.Model(data=model_data)
#
# print(model.n_subs)
#
# model.save(filepath=os.path.join('/Users/lucyowen/Desktop/analysis/ave_model/pyFR_20mm'))
# # model.save(filepath=os.path.join('/dartfs-hpc/scratch/lowen/ave_model/pyFR_20mm'))