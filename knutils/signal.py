import numpy as np
from scipy.signal import csd

def xwelch(x, **kwargs):
    f, __ = csd(x[:,0], x[:,0], **kwargs)
    cpsd = np.zeros([x.shape[1], x.shape[1], len(f)]).astype('complex')

    for i, xi in enumerate(x.T):
        for j, xj in enumerate(x.T):
            f, cpsd[i,j,:] = csd(xi, xj, **kwargs)

    return f, cpsd

def xfft(x, fs=1.0, onesided=True):
    n_samples = x.shape[0]
    f = np.fft.fftfreq(n_samples, 1/fs)
    cpsd = np.zeros([x.shape[1], x.shape[1], len(f)]).astype('complex')

    for i, xi in enumerate(x.T):
        Xi = np.fft.fft(xi)
        for j, xj in enumerate(x.T):
            Xj = np.fft.fft(xj) 
            cpsd[i,j,:] = 1/(fs*n_samples)*np.conj(Xi)*Xj        

    if onesided:
        n_sel = int(np.floor(n_samples/2))
        f = f[:n_sel]
        cpsd = 2*cpsd[:,:,:n_sel]
    return f, cpsd