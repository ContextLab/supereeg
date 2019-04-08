import supereeg as se
import numpy as np
import PyEMD
import matplotlib.pyplot as plt
import scipy

# simulate locations
locs = se.simulate_locations(n_elecs=3)

# simulate brain object
bo = se.simulate_bo(n_samples=500, sample_rate=100, cov='random', locs=locs, noise =.1)

# get numpy array of data, transposed so that each row is a timeseries of electrode data
data = bo.data.values.T

(n, m) = data.shape
time_points = np.multiply(1 / bo.sample_rate[0], np.arange(0, m))

EMD = PyEMD.EMD()
imfs_by_electrode = []
for i in range(0, n):
    imfs = EMD.emd(data[i], time_points)
    imfs_by_electrode.append(imfs)

for i in range(0, n):
    plt.figure(figsize=(14,14))
    currplot = 1
    imfs = imfs_by_electrode[i]
    plt.subplot(len(imfs) + 1, 1, currplot)
    currplot += 1
    plt.plot(time_points, data[i])
    plt.title('Electrode ' + str(i))
    plt.xlabel('Time [s]')
    for n, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, currplot)
        currplot += 1
        plt.plot(time_points, imf, 'g')
        plt.title('IMF ' + str(n+1))
        plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('hht/imfs_electrode' + str(i))
    plt.figure()
    plt.title('phase, electrode ' + str(i))
    plt.xlabel('Time [s]')
    for n, imf in enumerate(imfs):
        plt.plot(time_points, np.unwrap(np.angle(scipy.signal.hilbert(imf))), label='IMF %s' %(n+1))
    plt.legend(loc='upper right')
    plt.savefig('hht/phase_electrode' + str(i))    
    plt.figure()
    plt.title('frequency, electrode ' + str(i))
    plt.xlabel('Time [s]')
    dt = time_points[1] - time_points[0]
    for n, imf in enumerate(imfs):
        plt.plot(time_points[0:499], np.diff(np.unwrap(np.angle(scipy.signal.hilbert(imf)))/(2*np.pi*dt)), label='IMF %s' %(n+1))
    plt.legend(loc='upper right')
    plt.savefig('hht/freq_electrode' + str(i))

# plt.show()