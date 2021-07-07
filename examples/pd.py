import supereeg as se
import numpy as np
import scipy
import sys
import sklearn
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    fname = sys.argv[1]

# try:
bo = se.load(fname)
data = bo.data.T

delta = (2, 4)
theta = (4, 8)
alpha = (8, 12)
beta = (12, 30)
lgamma = (30, 60)
hgamma = (60, 100)
bands = [delta, theta, alpha, beta, lgamma, hgamma]
bands = [alpha]

HR = sklearn.linear_model.HuberRegressor()
for i, band in enumerate(bands):
    nyq = np.floor(bo.sample_rate[0] / 2)
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = scipy.signal.butter(2, [low, high], btype='band')
    band_passed = scipy.signal.filtfilt(b, a, data)
    hilberted = scipy.signal.hilbert(band_passed)
    print(len(hilberted[0]))
    log_power = np.log(np.abs(hilberted)**2)
    epoch = bo.sample_rate[0] * 2
    log_fftfreqs = np.log(np.fft.fftfreq(epoch, 1/bo.sample_rate[0]))
    
    for electrode in range(len(log_power)):
        slope = []
        intercept = []
        for time in list(range(len(log_power[0])))[::epoch]:
            fft = scipy.fftpack.fft(data[electrode][time:time+epoch])
            HR.fit(log_fftfreqs.reshape(-1, 1), fft)
            slope.append(HR.coef_[0])
            intercept.append(HR.intercept_)
        sl_mean = np.mean(slope)
        sl_sigma = np.std(slope)
        in_mean = np.mean(intercept)
        in_sigma = np.std(intercept)
    