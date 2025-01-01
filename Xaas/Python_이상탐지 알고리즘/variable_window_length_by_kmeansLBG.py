import numpy as np
from standardization_data import standardization_data
from kmeanlbg import kmeanlbg
from bayes_information_criterion import bayes_information_criterion


def variable_window_length_by_kmeansLBG(X1, PC, SD_type):
    k = PC['k_cluster']
    
    if X1['f0'].shape[0] > X1['L_max']:
        X1['f0'] = X1['f0'][-X1['L'][-1]:, :]
        
    numobs = X1['f0'].shape[0]
    # print(numobs)
    
    X = standardization_data(X1['f0'], X1['mu0'][-1, :], X1['sds0'][-1, :], numobs, SD_type)
    # print(len(X))
    
    IDX = []
    CENTROID = {}
    BIC = []
 
    for i in range(len(k)):
        ctrs, esq, idx = kmeanlbg(X, k[i])
     
        IDX.append(idx)
        CENTROID[i] = ctrs
        bic = bayes_information_criterion(X, idx, ctrs)
        BIC.append(bic)
    
    A, ind = max(BIC), np.argmax(BIC)
    idx = IDX[ind]
    
    n_cluster = k[ind]

    L = len(np.where(idx == idx[-1])[0])

    X1['N_cluster'].append(n_cluster)
    X1['L'] = np.append(X1['L'], L)
   
    X1['BIC'].append(BIC)
    
    if 'L0' in X1:
        X1['L0'] = np.append(X1['L0'], X1['L'][-1])   

    return X1