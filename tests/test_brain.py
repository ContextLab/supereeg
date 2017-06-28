# -*- coding: utf-8 -*-

import pytest
import superEEG as se
import numpy as np

def test_create_bo():
    data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)
    locs = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
    bo = se.Brain(data=data, locs=locs)
    assert isinstance(bo, se.Brain)
