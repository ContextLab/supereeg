import supereeg as se
import numpy as np
import scipy
import sys
import pycwt as wavelet
import matplotlib.pyplot as plt
import sklearn

if __name__ == "__main__":
    fname = sys.argv[0]

try:
    bo = se.load(fname)

    freqs = np.logspace(np.log10(2), np.log10(100))
    delta = freqs[0:8]
    theta = freqs[8:17]
    alpha = freqs[17:22]
    beta = freqs[22:33]
    lgamma = freqs[33:42]
    hgamma = freqs[42:50]

    epoch = 1000
    log_freqs = np.log(freqs)

    peak_deviations = np.zeros(shape=bo.data.T.shape)

    for electrode in range(0, len(bo.data.T)):
        wav_transform, sj, wavelet_freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][time:time+epoch], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
        raw_power = np.square(np.abs(wav_transform))
        avg_power = np.average(raw_power, axis=0)
        log_power = np.log(avg_power)
        
        HR = sklearn.linear_model.HuberRegressor()
        HR.fit(log_freqs.reshape(-1,1), log_power)
        narrowband_power = log_power - (log_freqs * HR.coef_[0] + HR.intercept_)
        peak_deviations[electrode] = narrowband_power

    deviation_bo = se.Brain(data=peak_deviations.T, locs=bo.locs, sample_rate=bo.sample_rate, filter=None)
    deviation_bo.save('peakdev_' + fname)

except:
    print('.bo file not found')
