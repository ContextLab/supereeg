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
# else:
    # simulate bo
locs = se.simulate_locations(n_elecs=3)
bo = se.simulate_bo(n_samples=3000, sample_rate=200, cov='random', locs=locs, noise =.1)

freqs = np.logspace(np.log10(2), np.log10(100))
delta = freqs[0:8]
theta = freqs[8:17]
alpha = freqs[17:22]
beta = freqs[22:33]
lgamma = freqs[33:42]
hgamma = freqs[42:50]


for time in list(range(0, 1)): #len(z_power[0])))[::1000]:
    wav_transform, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[0][time:time+1000], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
    raw_power = np.square(np.abs(wav_transform))
    log_power = np.log(raw_power)
    z_power = np.apply_over_axes(scipy.stats.mstats.zscore, raw_power, 1)
    for electrode in range(1, len(bo.data.T)):
        transformed, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][time:time+1000], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
        raw_power2 = np.square(np.abs(transformed))
        log_power2 = np.log(raw_power2)
        z_power2 = np.apply_over_axes(scipy.stats.mstats.zscore, raw_power2, 1)
        z_power += z_power2
    z_power /= len(bo.data.T)
    print(z_power.shape)
    HR = sklearn.linear_model.HuberRegressor()
    avg_power = np.average(z_power, axis=1)
    HR.fit(freqs.reshape(-1,1), avg_power)
    broadband_power = avg_power - (freqs * HR.coef_[0] + HR.intercept_)
    xs = np.linspace(0, 100, 500)
    ys = HR.coef_[0] * xs + HR.intercept_
    peak_deviation = np.amax(broadband_power)
plt.figure()
plt.plot(xs,ys)
plt.plot(freqs, avg_power)
plt.plot(freqs, broadband_power)



# for i in range(0, len(transformed)):
#     plt.figure()
#     plt.plot(np.arange(0,len(transformed[i])), transformed[i])
#     plt.title('freq ' + str(i))
# plt.show()
