# -*- coding: utf-8 -*-
"""
=============================
Resampling
=============================

This example shows you how to resample your data

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_data')

# info contains sample rate
bo.info()

# default resample to 64Hz
bo_d = bo.get_resampled()

# show new info
bo_d.info()

# resample to specified sample rate
bo_n = bo.get_resampled(100)

# show new info
bo_n.info()







