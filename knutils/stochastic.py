from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm

def harmonic_spectrum(omega_peak, amplitude, domega, omega_max, as_variance=None, minimum_duration=0):
    if minimum_duration!=0:
        domega = 2*np.pi/minimum_duration

    nfft = int(2**nextpow2(np.ceil(omega_max/domega)))
    domega = omega_max/nfft

    if as_variance:
        variance = amplitude
    else:
        variance = amplitude**2/2

    omega = np.hstack([np.flip(np.arange(omega_peak, 0, -domega), axis=0),np.arange(omega_peak+domega, omega_max, domega)])
    S = omega*0.0
    S[omega==omega_peak] = variance/domega
        
    return omega, S

    
def rayleigh_variance_to_scaling(var):
    scaling = np.sqrt(2*var/(4 - np.pi))
    return scaling


def random_phase_angles(n_len, n_sim=1):
    phase_angles = np.random.rand(n_len, n_sim)*2*np.pi    
    if n_sim==1:
        phase_angles = phase_angles[:, 0:1]

    return phase_angles    


def random_amplitude(n_len, n_sim=1, var=1.0):
    amplitude = np.random.rayleigh(scale=rayleigh_variance_to_scaling(var), size=[n_len, n_sim])/2.0    #verify why this should be 2.0 (works with var=1.0 and montecarlo...)

    return amplitude


def montecarlo(omega, S, phase_angles, amplitude_scaling=None, ifft=True):  
    # Input must be such that omega[0] refers to static (freq=0) condition.
    domega = omega[1] - omega[0] 
    c = np.sqrt(2*S*domega)

    if amplitude_scaling is not None:
        c = amplitude_scaling*c

    if ifft==True:
        f = len(omega)*np.real(np.fft.ifft(c*np.exp(1j*phase_angles)))
    else:
        f = np.real(np.fft.fft(c*np.exp(-1j*phase_angles)))

    return f


def fft_time(omega, t0=0):
    domega = omega[1]-omega[0] 
    t = np.linspace(t0, np.pi*2/domega, len(omega))

    return t


def adjust_for_ifft(S, omega, duration=None, samplerate=None, nfft_roundup=True, interpol_kind='linear', avoid_warnings=False):  
    # For extremely narrow-banded S (e.g. single harmonics) do not specify duration without extreme caution - this might cause problems for the discretization of S. 
    # #The same goes for the nfft_roundup flag (should only be used with extreme caution)

    domega = omega[2] - omega[1]
   
    if duration is None and samplerate is None:
        if omega[0]!=0.0:
            omega_adjusted = np.arange(0, max(omega), domega)

        else:
            omega_adjusted = 0+omega
            S_adjusted = 0+S
    else:
        if duration is None:
            duration = np.pi*2/domega

        if samplerate is None:
            samplerate = np.max(omega)/2/np.pi

        df = 1/duration

        if df*2*np.pi>domega and not avoid_warnings:
            print('WARNING: The specified duration gives a coarser frequency resolution than the inputted spectral density. This could cause problems.')

        if samplerate*2*np.pi<max(omega) and not avoid_warnings:
            print('WARNING: The specified samplerate gives a maximum frequency below the specified frequency axis.')

        omega_adjusted = np.arange(0, samplerate*2*np.pi, df*2*np.pi)

    if nfft_roundup is True:
        nfft = int(2**nextpow2(len(omega_adjusted)))
        omega_adjusted = np.linspace(0, max(omega_adjusted),nfft)
    
    S_adjusted = interp1d(omega, S, kind=interpol_kind, fill_value=0, bounds_error=False)(omega_adjusted)
    S_adjusted[S_adjusted<0] = 0    #remove negative values (truncate to 0)

    return S_adjusted, omega_adjusted

    
def nextpow2(a):
    exponent = np.ceil(np.log2(abs(a)))
    return exponent


def flat_spectrum(S0, spectrum_range, omega, is_variance=False):

    if spectrum_range[1]>max(omega) or spectrum_range[0]<min(omega):
        raise ValueError('Range outside omega.')

    S = omega*0
    startix = np.where(omega<=spectrum_range[0])[0][-1]
    stopix = np.where(omega>=spectrum_range[1])[0][0]

    if is_variance:
        varS = S0*1.0
        S0 = varS/(spectrum_range[1]-spectrum_range[0])

    S[startix:stopix+1] = S0
    
    return S
   

def get_short_term_prob(S, omega, T):
    from wawi.random import v0_from_spectrum
    v0_max = np.max(np.diag(np.real(v0_from_spectrum(S, omega))))
    Nmax = v0_max*T
    p = 1/Nmax
    
    return p

def short_term_iform_from_cpsd(S, omega, T, n_points=1000):
    from wawi.random import v0_from_spectrum
    R = np.trapz(np.real(S), omega, axis=-1)
    p = get_short_term_prob(S, omega, T)
    
    return short_term_iform_from_cov(R, p, n_points=100)
    
def get_extreme_prob(S, omega, T):
    from scipy.stats import norm
    from wawi.random import expmax_from_cpsd, expmax_from_spectrum
    beta_extreme=0.0
    var = np.diag(np.trapz(S, x=omega))
    std = np.real(np.sqrt(var))
    
    for comp in range(S.shape[0]):
        beta_extreme = np.max([beta_extreme, np.real(expmax_from_spectrum(S[comp,comp,:], omega, T))]/std[comp])
    p_extreme = 1 - norm.cdf(beta_extreme)
    
    return p_extreme

def short_term_iform_from_cov(R, p, n_points=100):
    """
    Calculates a 2D load contour using IFORM for a given return period.


    Parameters
    ----------
    R : ndarray
        covariance matrix for the components
    p : float
        target probability
    n_points : int, optional
        Number of points to be calculated on the contour.

    Returns
    -------
    ndarray
        An (n_points x 2) array with load combinations [X1, X2] on the contour.
    """

    n_dims = R.shape[0]
    
    # Steg 1: Bestem målsannsynlighet (beta)
    beta = -norm.ppf(p)

    # Steg 2: Definer konturen i n-dimensjonalt u-rom (hypersfære)
    random_points = np.random.normal(size=(n_dims, n_points))
    norms = np.linalg.norm(random_points, axis=0)
    u_hypersphere = beta * random_points / norms
    # ------------------------------------

    # Steg 3: Transformer konturen til fysisk rom (x-rom)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError('R is not positive definite.')

    x_contour = L @ u_hypersphere
    
    return x_contour.T # Returnerer (n_points x n_dims)