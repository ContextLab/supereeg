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

    bands = [delta, theta, alpha, beta, lgamma, hgamma]

    power = np.zeros(shape=(bo.data.T.shape[1], 6000))

    for electrode in range(0, len(bo.data.T)):
        #not full brain object!
        wav_transform, sj, wavelet_freqs, coi, fft, fftfreqs = wavelet.cwt(bo.data[electrode][36000:42000], 1/bo.sample_rate[0], freqs = freqs, wavelet=wavelet.Morlet(4))
        raw_power = np.square(np.abs(wav_transform))
        avg_power = np.average(raw_power, axis=0)
        power[electrode] = avg_power

    power_bo = se.Brain(data=power.T, locs=bo.locs, sample_rate=bo.sample_rate, filter=None)
    power_bo.save('power_'+fname)

except:
    print('.bo file not found')
