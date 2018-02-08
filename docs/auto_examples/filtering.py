# -*- coding: utf-8 -*-
"""
=============================
Filtering electrodes and subjects
=============================

This example filters electrodes based on kurtosis thresholding.
It also filters patients if less than two electrodes pass thresholding.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se
import os

# load example data
subject_1 = os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/BW013.bo'
bo = se.load(subject_1)

# filter elecs, default measure='kurtosis' and threshold=10
f_bo = se.filter_elecs(subject_1)

