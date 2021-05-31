import numpy as np

def increase(x, dx):
    x0 = int(np.floor(x/dx))
    intervals = np.array([x0]*dx)
    
    is_symm = (x-1) % 2 == dx % 2
    is_odd = dx%2 != 0
    
    print(is_symm)
    print(is_odd)
    
    rem = x - np.sum(intervals)

    if rem != 0:
        n = int(np.ceil(dx/rem))
        xc = int(np.ceil((dx-1)/2))
        
        intervals[xc-1+is_symm::-n] += 1 
        
        if np.sum(intervals)<x:
            intervals[xc+is_symm::n] += 1
        
        if np.sum(intervals)<x:
            rem2 = x-np.sum(intervals)
            min_ix = np.where(intervals == np.min(intervals))[0]
    
            med_ix = np.mean(min_ix)
            min_ix = min_ix[np.argsort(np.abs(min_ix - med_ix))]
            intervals[min_ix[:rem2]] += 1
    
    
    return intervals

x = 19
dx = 12

test = increase(x,dx)
print(sum(test))
print(test)