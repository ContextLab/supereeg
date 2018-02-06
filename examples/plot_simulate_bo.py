# -*- coding: utf-8 -*-
"""
=============================
Simulate brain object
=============================

In this example, we demonstrate the simulate brain object function.
First, we'll load in some example locations. Then we'll simulate 1
brain object specifying a noise parameter and the correlational structure
(a toeplitz matrix).
"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

import supereeg as se
import pandas as pd

# load example model to get locations
locs = se.load('example_locations')

# convert to pandas
locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

# simulate brain object
bo = se.simulate_bo(n_samples=100, sample_rate=1000, cov='toeplitz', locs=locs, noise =.3)

