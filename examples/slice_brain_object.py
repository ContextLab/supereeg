# -*- coding: utf-8 -*-
"""
=============================
Slice brain object
=============================

Here, we load an example dataset, and then slice out some electrodes and time
samples.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_data')

# check out the brain object (bo)
bo.info()

# index by first 5 timepoints
bo_t = bo.get_slice(sample_inds=range(5))

# or index by the 10th location
bo_l = bo.get_slice(loc_inds=10)

# or index by both locations and times
bo_i = bo.get_slice(sample_inds=[1,2,3,4,5], loc_inds=[10,11,12])

bo_i.info()
