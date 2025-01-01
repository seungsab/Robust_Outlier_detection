import numpy as np
from disteusq import disteusq


def k_means(d, k, x0=None):
    
    n, _ = d.shape

    if x0 is None:
        x = d[np.random.choice(n, k, replace=False), :]
        
    else:
        x = x0

    y = x + 1
   
    iter = 0
    while not np.all(x == y):
        iter += 1
        
        z = disteusq(x=d, y=x, mode='x')
    
        m = np.min(z, axis=1)
        j = np.argmin(z, axis=1)
      
        y = x.copy()

        for i in range(k):
            
            s = j == i
            if np.any(s):
                x[i, :] = np.mean(d[s, :], axis=0)
            else:
                q = np.where(m != 0)[0]
                if len(q) == 0:
                    break
                r = np.random.choice(q)
                x[i, :] = d[r, :]
                m[r] = 0

                y = x.copy() + 1

    esq = np.mean(m)
    
    return x, esq, j