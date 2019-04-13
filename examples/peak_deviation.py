import supereeg as se
import numpy as np
import scipy
import sys
import pycwt as wavelet
import matplotlib.pyplot as plt
import sklearn

# if __name__ == "__main__":
#     fname = sys.argv[0]

# if fname != '':
#     bo = se.load(fname)

bo = se.load('P0010sz1_seizure.bo')
# bo = se.load('example_data')
# locs = se.simulate_locations(n_elecs=10)
# bo = se.simulate_bo(n_samples=3000, sample_rate=200, cov='random', locs=locs, noise =.1)

freqs = np.logspace(np.log10(2), np.log10(100))
delta = freqs[0:8]
theta = freqs[8:17]
alpha = freqs[17:22]
beta = freqs[22:33]
lgamma = freqs[33:42]
hgamma = freqs[42:50]

epoch = 100

time_points, num_electrodes = bo.data.values.shape
peak_deviations = []

for time in list(range(0, len(bo.data[0])))[0:1000:epoch]:
    wav_transform, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[0][time:time+epoch], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
    raw_power = np.square(np.abs(wav_transform))
    avg_power = np.average(raw_power, axis=1)
    for electrode in range(1, len(bo.data.T)):
        transformed, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][time:time+epoch], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
        avg_power += np.average(np.square(np.abs(transformed)), axis=1)
    
    avg_power /= len(bo.data.T)
    log_power = np.log(avg_power)
    z_power = scipy.stats.mstats.zscore(log_power)
    log_freqs = np.log(freqs)
    HR = sklearn.linear_model.HuberRegressor()
    HR.fit(log_freqs.reshape(-1,1), log_power)
    narrowband_power = log_power - (log_freqs * HR.coef_[0] + HR.intercept_)
    xs = np.linspace(0, 5, 100)
    ys = HR.coef_[0] * xs + HR.intercept_
    peak_deviation = np.amax(narrowband_power)
    peak_deviations.append(peak_deviation)
    plt.figure()
    plt.plot(xs,ys, label='broad')
    plt.plot(log_freqs, log_power, label='log')
    plt.plot(log_freqs, narrowband_power, label='narrow')


deviation_bo = se.Brain(data=peak_deviations, locs=bo.locs, sample_rate=bo.sample_rate)
deviation_bo.plot_data()
deviation_bo.plot_locs()
# bo.plot_data()
# plt.figure()
# plt.plot(xs,ys, label='broad')
# plt.plot(log_freqs, log_power, label='log')
# plt.plot(log_freqs, narrowband_power, label='narrow')
# plt.legend(loc='upper right')
# plt.figure()
# plt.plot(freqs, raw_power)
plt.show()
