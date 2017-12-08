
import superEEG as se
import numpy as np
from superEEG._helpers.stats import tal2mni
import glob
import sys
import os
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

results_dir = os.path.join(config['resultsdir'], model_template)

try:
    os.stat(results_dir)
except:
    os.makedirs(results_dir)



# load locations for model
### this weird work around is necessary because there's an issue using a variable for a string in an argument

if model_template == 'mini_model_nifti':
    gray = se.load('mini_model_nifti')

elif model_template == 'pyFR_union':
    gray = se.load('pyFR_union')

elif model_template == 'big_temp':
    gray = se.load('big_temp')
else:
    gray = se.load('mini_model_nifti')



# extract locations
gray_locs = gray.locs

file_name = os.path.basename(os.path.splitext(fname)[0])

# work around if not brain objects, but :
if fname.split('.')[-1]=='bo':
    bo = se.load(fname)

else:
    print('unknown file type')


model = se.Model(bo, locs = gray_locs)

model.save(filepath=os.path.join(results_dir, file_name))


print('done')

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