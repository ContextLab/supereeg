# -*- coding: utf-8 -*-

# load
import supereeg as se
import numpy as np

# load example data
bo = se.load('peakdev.bo')

# simulate 100 locations
# locs = se.simulate_locations(n_elecs=100)

# simulate brain object
# bo = se.simulate_bo(n_samples=400, sample_rate=100, cov='random', locs=locs, noise =.1)

# convert to nifti
nii = bo.to_nii(template='std', vox_size=6)

# make gif
# '/your/path/to/gif/'
nii.make_sliced_gif('C:\\Users\\tmunt\\Documents\\gif', time_index=np.arange(len(bo.data[0])), slice_index=range(-4,52,4), name='sample_gif', vmax=3, symmetric_cbar=False, duration=200, alpha=0.7)
