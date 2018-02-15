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

bo.get_resampled_data()




