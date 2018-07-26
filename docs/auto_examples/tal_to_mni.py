# -*- coding: utf-8 -*-
"""
=============================
Convert from talairach to MNI space
=============================

This example converts electrodes locations from talairach to MNI space.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se
import numpy as np

# some example electrode locations
tal_locs = np.array([[-54, -9, -15], [-54, -5, -7], [-52, -1, 2]])

# convert to mni space
mni_locs = se.tal2mni(tal_locs)
