import numpy as np
from k_means import k_means

def kmeanlbg(d, k):
    nc = d.shape[1]
    
    x, esq, j = k_means(d, 1)

    m = 1
    iter = 0
    while m < k:
        iter += 1
        n = min(m, k - m)
        m = m + n
        
        e = 1e-4 * np.sqrt(esq) * np.random.rand(1, nc)
        
        x, esq, j = k_means(d, m, np.vstack([x[:n, :] + e[:n, :], x[:n, :] - e[:n, :], x[n:m-n, :]]))
        
    return x, esq, j