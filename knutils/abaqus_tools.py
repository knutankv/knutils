import numpy as np
import matplotlib.pyplot as plt

def abq_def(t, A, dt=None):
    min_T = np.min(np.diff(t))
    if dt is None:
        dt = min_T/20
        
    t_full = np.arange(t[0], t[-1], dt)
    A_full = t_full*0

    
    for i in range(len(t)-1):
        t1 = t[i]
        t2 = t[i+1]
        ok = (t_full>=t1) * (t_full<=t2)

        xi = (t_full[ok]-t1)/(t2-t1)
        A_full[ok] = A[i] + (A[i+1] - A[i])*xi**3 * (10 - 15*xi + 6*xi**2)        
    
    return t_full, A_full
