from __future__ import division
import supereeg as se
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supereeg.helpers import _z2r


locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21.,  63.,  -3.],
                 [ -1., -37.,  37.],
                 [ -1.,  23.,  17.],
                 [ 19., -57., -23.],
                 [ 19.,  23.,  -3.],
                 [ 39., -57.,  17.],
                 [ 39.,   3.,  37.],
                 [ 59., -17.,  17.]])


# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 6
# number of electrodes
n_elecs = 5
# simulate correlation matrix
data1 = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]
data2 = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = n_elecs) for x in range(n_subs)]

# test model to compare
m1 = se.Model(data=data1, locs=locs)
m2 = se.Model(data=data2, locs=locs)

c1 = m1 + m2
m2_recon = c1 - m1
m1_recon = c1 - m2
c1_recon = m1_recon + m2_recon

def compare_matrices(a, b, a_name, b_name, label):
    m_a = eval('a.' + label)
    m_b = eval('b.' + label)
    base_str = a_name + ' and ' + b_name + ' ' + label + 's '
    if np.allclose(m_a, m_b, equal_nan=True):
        print(base_str + 'match!')
    else:
        print(base_str + 'do NOT match...')

def compare_models(a, b, a_name, b_name):
    def compare_helper(x, y):
        if np.allclose(x, y, equal_nan=True):
            return 'match!'
        else:
            return 'NOT a match...'
    manual_a = _z2r(np.divide(a.numerator, a.denominator))
    manual_b = _z2r(np.divide(b.numerator, b.denominator))
    print('numerator ' + a_name + ' vs ' + b_name + ': ' + compare_helper(a.numerator, b.numerator))
    print('denominator ' + a_name + ' vs ' + b_name + ': ' + compare_helper(a.denominator, b.denominator))
    print('manual ' + a_name + ' vs ' + b_name + ': ' + compare_helper(manual_a, manual_b))
    print('auto ' + a_name + ' vs ' + b_name + ': ' + compare_helper(a.get_model(), b.get_model()))


models = ['m1', 'm2', 'c1']
fields = ['numerator', 'denominator', 'get_model()']

for a_name in models:
    b_name = a_name + '_recon'
    for f in fields:
        compare_matrices(eval(a_name), eval(b_name), a_name, b_name, f)

print('\n\n')

for m in models:
    n = m+'_recon'
    compare_models(eval(m), eval(n), m, n)
    print('\n')

c2 = c1 + c1