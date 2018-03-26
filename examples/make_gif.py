# -*- coding: utf-8 -*-
"""
=============================
Make gif
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.  We then convert the reconstruction to a nifti and plot 3 consecutive timepoints
first with the plot_glass_brain and then create .png files and compile as a gif.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# load
import supereeg as se


# load example data
bo = se.load('example_data')

# load example model
model = se.load('example_model')

# the default will replace the electrode location with the nearest voxel and reconstruct at all other locations
reconstructed_bo = model.predict(bo)

# print out info on new brain object
reconstructed_bo.info()

# slice first 3 timepoints
samples = [0,1,2]
reconstructed_bo.get_slice(sample_inds=samples, inplace=True)



# convert to nifti
reconstructed_nifti = reconstructed_bo.to_nii(template='gray', vox_size=20)

# plot first 5 timepoints
reconstructed_nifti.plot_glass_brain(index=samples)

# make gif, default time window is 0 to 10, but you can specifiy by setting a range with index
# reconstructed_nifti.make_gif(gif_path='/your/path/to/gif', index=samples, name='sample_gif')
