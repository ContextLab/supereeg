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
bo.resample()

# show new info - nothing changed if resample_rate isn't specified
bo.info()

# resample to specified sample rate
bo.resample(64)

# show new info
bo.info()







