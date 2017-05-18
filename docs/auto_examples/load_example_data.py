# -*- coding: utf-8 -*-
"""
=============================
Loading data
=============================

Here, we load an example dataset and then print out some information about it.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG

# load example data
bo = superEEG.load_example_data()

# check out the bo
bo.info()
