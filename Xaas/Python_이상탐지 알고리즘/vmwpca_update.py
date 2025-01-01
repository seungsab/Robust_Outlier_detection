import numpy as np
from standardization_data import standardization_data
from variable_window_length_by_kmeansLBG import variable_window_length_by_kmeansLBG
from select_retained_nPC import select_retained_nPC
from scipy.stats import chi2
from sklearn.decomposition import PCA


def VMWPCA_update(X0, PC, Q, alpha, SD_type, nPC_select_algorithm, ind_x):
    f, mu0, sds0 = X0['f'], X0['mu0'], X0['sds0']
  
    loadings, n_pc, variances = PC['loadings'], PC['n_pc'], PC['variances']
    Qdist, distcrit = Q['Qdist'], Q['distcrit']
    
    numobs1 = 1
    X1std = standardization_data(f[-1, :], mu0[-1, :], sds0[-1, :], numobs1, SD_type)

    scores1 = np.dot(X1std, loadings[:, :n_pc[-1]])
    residuals_val = (X1std - np.dot(scores1[:, :n_pc[-1]], loadings[:, :n_pc[-1]].T))
    Qdist_val = np.sqrt(np.sum(residuals_val ** 2, axis=1))
    Qdist = np.append(Qdist, Qdist_val)

    if alpha == '95%':
        Q_lim = distcrit[-PC['n_stall']:, 0]
    elif alpha == '99%':
        Q_lim = distcrit[-PC['n_stall']:, 1]

    Q_check = Qdist[-PC['n_stall']:] <= Q_lim
    
    Normal_condition = 0
    if Q_check[-1] == 0:
        PC['outlier'] = np.append(PC['outlier'], ind_x)
        PC['fault'] = np.append(PC['fault'], PC['fault'][-1] + 1)
    else:
        Normal_condition = 1
        PC['fault'] = np.append(PC['fault'], 0)
    
    if PC['fault'][-1] > PC['n_stall']:
        PC['update'] = 0

    if PC['update']:
        if Normal_condition == 1:
            X0['f0'] = np.vstack([X0['f0'], f[-1, :]])
      
            X0 = variable_window_length_by_kmeansLBG(X0, PC, SD_type)
 
            f = X0['f0'][-X0['L'][-1]:, :]
        
            n = f.shape[0]
        
            mu0 = np.vstack([mu0, np.mean(f, axis=0)])
            sds0 = np.vstack([sds0, np.std(f, axis=0)])
            numobs = f.shape[0]
          
            X0std = standardization_data(f, mu0[-1, :], sds0[-1, :], numobs, SD_type)
            
            pca = PCA()
            pca.fit(X0std)
            loadings = pca.components_.T
            scores = pca.transform(X0std)
            variances = pca.explained_variance_
            
            n_pc1 = select_retained_nPC(variances, nPC_select_algorithm)
            n_pc = n_pc + n_pc1
            
            
            residuals = X0std - np.dot(scores[:, :n_pc[-1]], loadings[:, :n_pc[-1]].T)
            
            Qdist1 = np.sqrt(np.sum(residuals ** 2, axis=1))
            
            m_Q = np.mean(Qdist1)
            V_Q = np.var(Qdist1)
            V = 2 * (m_Q ** 2) / V_Q
            
            distcrit1 = V_Q / (2 * m_Q) * np.array([
                chi2.ppf(0.95, V),
                chi2.ppf(0.99, V)
            ])
            
            distcrit = np.vstack([distcrit, distcrit1])
            
            PC['loadings'], PC['n_pc'], PC['variances'] = loadings, n_pc, variances

        else:
            f = f[:-1, :]
            n = f.shape[0]

            mu0 = np.vstack([mu0, np.mean(f, axis=0)])
            sds0 = np.vstack([sds0, np.std(f, axis=0)])
            numobs = f.shape[0]
            n_pc = n_pc.append(n_pc[-1])

            distcrit = np.vstack([distcrit, distcrit[-1, :]])
            X0['L0'] = np.append(X0['L0'], X0['L'][-1])

    else:
        f = f[:-1, :]
        n = f.shape[0]

        mu0 = np.vstack([mu0, np.mean(f, axis=0)])
        sds0 = np.vstack([sds0, np.std(f, axis=0)])
        numobs = f.shape[0]
        n_pc = n_pc.append(n_pc[-1])

        distcrit = np.vstack([distcrit, distcrit[-1, :]])
        X0['L0'] = np.append(X0['L0'], X0['L'][-1])

    X0['f'], X0['mu0'], X0['sds0'] = f, mu0, sds0
    Q['Qdist'], Q['distcrit'] = Qdist, distcrit

    return X0, PC, Q