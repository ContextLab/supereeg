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

# indexing:

#returns first time sample
bo1 = bo[0]

#return first 5 time samples
bo2 = bo[:5]

#return first electrode
bo3 = bo[:, 0]

#returns first 5 timesamples/elecs
bo4 = bo[:5, :5]

# or index by both locations and times in place using get_slice method and specify inplace=True
bo.get_slice(sample_inds=[0,1,2,3,4], loc_inds=[0,1,2,3,4], inplace=True)
