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


bo = se.load('P0010sz1.bo')

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
    avg_power = np.average(raw_power, axis=1)
    plt.figure()
    plt.plot(freqs, avg_power)
    for electrode in range(1, len(bo.data.T)):
        transformed, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][time:time+1000], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
        avg_power += np.average(np.square(np.abs(transformed)), axis=1)
    avg_power /= len(bo.data.T)
    log_power = np.log(avg_power)
    print(log_power.shape)
    z_power = scipy.stats.mstats.zscore(log_power)
    print(z_power.shape)
    HR = sklearn.linear_model.HuberRegressor()
    HR.fit(freqs.reshape(-1,1), z_power)
    narrowband_power = z_power - (freqs * HR.coef_[0] + HR.intercept_)
    xs = np.linspace(0, 100, 500)
    ys = HR.coef_[0] * xs + HR.intercept_
    peak_deviation = np.amax(narrowband_power)


plt.figure()
plt.plot(xs,ys, label='broad')
plt.plot(freqs, avg_power, label='avg')
plt.plot(freqs, narrowband_power, label='narrow')
plt.legend(loc='upper right')
plt.figure()
plt.plot(freqs, raw_power)
plt.show()
