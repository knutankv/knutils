import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft, fftfreq

def coherency_from_cpsd(S):
    gamma = S*0.0
    for comp1 in range(S.shape[0]):
        for comp2 in range(S.shape[0]):
            gamma[comp1, comp2,:] = S[comp1, comp2]/np.sqrt(S[comp1, comp1, :] * S[comp2, comp2, :])
            
    return gamma

def xwelch(x, **kwargs):
    f, __ = csd(x[:,0], x[:,0], **kwargs)
    cpsd = np.zeros([x.shape[1], x.shape[1], len(f)]).astype('complex')

    for i, xi in enumerate(x.T):
        for j, xj in enumerate(x.T):
            f, cpsd[i,j,:] = csd(xi, xj, **kwargs)

    return f, cpsd

def xfft(x, fs=1.0, onesided=True, **kwargs):
    n_samples = x.shape[0]
    f = np.fft.fftfreq(n_samples, 1/fs)
    cpsd = np.zeros([x.shape[1], x.shape[1], len(f)]).astype('complex')

    for i, xi in enumerate(x.T):
        Xi = np.fft.fft(xi,**kwargs)
        for j, xj in enumerate(x.T):
            Xj = np.fft.fft(xj,**kwargs) 
            cpsd[i,j,:] = 1/(fs*n_samples)*np.conj(Xi)*Xj        

    if onesided:
        # n_sel = int(np.floor(n_samples/2))
        # f = f[:n_sel]
        # cpsd = 2*cpsd[:,:,:n_sel]
        
        f = f[0:n_samples//2]
        cpsd = 2*cpsd[:,:,0:n_samples//2]
        
    return f, cpsd

def time_int(x, fs, levels=1, apply_filter=None):
    L = x.shape[0]
    dt = 1/fs
    f = fftfreq(L, dt)
    x_int = [x] + [None]*levels
    for level in range(1,levels+1):
            this_x_fft = (2*np.pi*f*1j)**(-1) * fft(x_int[level-1])
            this_x_fft[0] = 0
            x_int[level] = np.real(ifft(this_x_fft))

            if apply_filter is not None:
                x_int[level] = apply_filter(x_int[level])

    return x_int


def ramp_up(Nramp, Ntot):
    t_scale = np.ones(Ntot)
    t_scale[:Nramp] = np.linspace(0, 1, Nramp)
    return t_scale

def ramp_up_t(t, t0):
    Nramp = np.sum(t<t0)
    Ntot = len(t)
    
    return ramp_up(Nramp, Ntot)