import supereeg as se
import numpy as np
import scipy
import sys
import pycwt as wavelet
import matplotlib.pyplot as plt
import sklearn

if __name__ == "__main__":
    fname = sys.argv[1]

try:
    bo = se.load(fname)

    freqs = np.logspace(np.log10(2), np.log10(100))
    delta = freqs[0:8]
    theta = freqs[8:17]
    alpha = freqs[17:22]
    beta = freqs[22:33]
    lgamma = freqs[33:42]
    hgamma = freqs[42:50]

    epoch = 400

    peak_deviations = np.zeros(shape=(bo.data.shape[1], int(np.floor(bo.data.shape[0]/epoch))))

    bands = [delta, theta, alpha, beta, lgamma, hgamma]

    for i, band in enumerate(bands):
        toprange = int(np.floor(len(bo.data[0])/epoch)*epoch)
        for time in list(range(0, toprange))[::epoch]:
            for electrode in range(0, len(bo.data.T)):
                wav_transform, sj, wavelet_freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][time:time+epoch], 1/bo.sample_rate[0], freqs = band, wavelet=wavelet.Morlet(4))
                raw_power = np.square(np.abs(wav_transform))
                avg_power = np.average(raw_power, axis=1)
                log_power = np.log(avg_power)
                log_freqs = np.log(band)
                HR = sklearn.linear_model.HuberRegressor()
                HR.fit(log_freqs.reshape(-1,1), log_power)
                narrowband_power = log_power - (log_freqs * HR.coef_[0] + HR.intercept_)
                m = np.amax(narrowband_power)
                peak_deviations[electrode][int(time/epoch)] = m
            # print('time ' + str(time) + ' of ' + str(toprange))


        deviation_bo = se.Brain(data=peak_deviations.T, locs=bo.locs, sample_rate=bo.sample_rate, filter=None)
        deviation_bo.save('pdpe_band' + str(i) +'_'+fname)

except:
    print('.bo file not found')
