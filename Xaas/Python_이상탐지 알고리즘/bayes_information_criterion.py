import numpy as np

def bayes_information_criterion(X, idx, ctrs):
    n, d = X.shape
    m = len(ctrs)

    BIC = 0
    for i in range(m):
        index_i = np.where(idx == i)[0]
        Xi = X[index_i, :]
        ni = Xi.shape[0]
        mi = np.mean(Xi, axis=0)
        
        sigma_i = 1 / (ni - m) * np.sum(np.sum((Xi - mi) ** 2, axis=1))
        
        BIC += ni * np.log(ni / n) - ni * d / 2 * np.log(2 * np.pi) - ni / 2 * np.log(sigma_i) - (ni - m) / 2
    
    BIC -= m * np.log(n) / 2

    if not np.isreal(BIC):
        print(f"Non-real BIC encountered: {BIC}")
    
    return BIC