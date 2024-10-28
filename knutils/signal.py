import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft, fftfreq

def coherency_from_cpsd(S):
    gamma = S*0.0
    for comp1 in range(S.shape[0]):
        for comp2 in range(S.shape[0]):
            gamma[comp1, comp2,:] = S[comp1, comp2]/np.sqrt(S[comp1, comp1, :] * S[comp2, comp2, :])
            
    return gamma

from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz
from scipy.integrate import cumulative_trapezoid

def butter_construct(cuts, fs, btype='band', order=5):
        nyq = 0.5 * fs
        cuts = [cut/nyq for cut in cuts]
        sos = butter(order, cuts, analog=False, btype=btype, output='sos')

        return sos


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


def time_integrate(data, fs, levels, domain='frequency', axis=0, filters=[]):

    for filter_i in filters:
        data = sosfilt(filter_i, data, axis=axis)
    
    datai = [None]*(levels+1)
    datai[0] = data*1
    
    
    for i in range(1,levels+1):
        if domain == 'frequency':
            f = np.fft.fftfreq(data.shape[0])
            
            fft_data = np.fft.fft(datai[i-1], axis=axis)
            factor = (1./(2*np.pi*f[1:]*1j))**i
            thisdataf = fft_data*0
            thisdataf[1:, :] = fft_data[1:, :]*factor
            datai[i] = np.real(np.fft.ifft(thisdataf, axis=axis))
        else:
            t = np.arange(0, data.shape[0]*(1/fs), 1/fs)
            datai[i] = cumulative_trapezoid(datai[i-1], x=t, axis=axis, initial=0.0)
    
    return datai[1:]

def ramp_up(Nramp, Ntot):
    t_scale = np.ones(Ntot)
    t_scale[:Nramp] = np.linspace(0, 1, Nramp)
    return t_scale

def ramp_up_t(t, t0):
    Nramp = np.sum(t<t0)
    Ntot = len(t)
    
    return ramp_up(Nramp, Ntot)
