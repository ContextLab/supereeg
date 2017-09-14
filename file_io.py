
import superEEG as se
import numpy as np
from superEEG._helpers.stats import tal2mni
import glob
import os



path_name = '/Users/lucyowen/Desktop/analysis/'
#path_name = '/idata/cdl/data/ECoG/pyFR/data'


#
# def npz2bo(infile):
#     with open(infile, 'rb') as handle:
#         f = np.load(handle)
#         f_name = os.path.splitext(os.path.basename(infile))[0]
#         data = f['Y']
#         sample_rate = f['samplerate']
#         sessions = f['fname_labels']
#         locs = tal2mni(f['R'])
#         meta = f_name
#
#     return se.Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta=meta)
#
#


##### for making brain objects from npz
# files = glob.glob(os.path.join(path_name,'npz/*.npz'))
# for i in files:
#     file_name = os.path.splitext(os.path.basename(i))[0]
#     bo = npz2bo(i)
#     bo.save(filepath=os.path.join(path_name + '/bo', file_name))
#
# print('done')


### to create model with all patients
#
# model_data = []
# bo_files = glob.glob(os.path.join(path_name,'bo/*.bo'))
#
# ### instead of loading each, can I just load one field??
#
# for b in bo_files:
#     model_data.append(se.filter_subj(se.load(b)))
#
#
# model = se.Model([se.load(os.path.join(path_name, 'bo', b + '.bo')) for b in model_data[:2]])
# for b in model_data[2:]:
#     if b == None:
#         continue
#     else:
#         model = model.update(se.load(os.path.join(path_name, 'bo', b + '.bo')))
# print(model.n_subs)
#
# #model.save(filepath=os.path.join('/Users/lucyowen/Desktop/analysis/ave_model/pyFR_20mm'))
#
# model.save(filepath=os.path.join('/dartfs-hpc/scratch/lowen/ave_model/pyFR_20mm'))


###### to create model with N-1 patient and reconstruct at that patient

model_data = []

bo_files = glob.glob(os.path.join(path_name,'bo/[!BW001]*.bo'))

### instead of loading each, can I just load one field??

for b in bo_files:
    model_data.append(se.filter_subj(se.load(b)))


model = se.Model([se.load(os.path.join(path_name, 'bo', b + '.bo')) for b in model_data[:2]])
for b in model_data[2:]:
    if b == None:
        continue
    else:
        model = model.update(se.load(os.path.join(path_name, 'bo', b + '.bo')))
print(model.n_subs)

file_name = 'example_pyFR_20mm'

#model.save(filepath=os.path.join('/Users/lucyowen/Desktop/analysis/ave_model', file_name))

model.save(filepath=os.path.join('/dartfs-hpc/scratch/lowen/ave_model', file_name))

BW001 = se.load('example_data')

reconstructed = model.predict(BW001)

reconstructed.save('example_recon_20mm')

reconstructed.to_nii('example_recon_20mm')


