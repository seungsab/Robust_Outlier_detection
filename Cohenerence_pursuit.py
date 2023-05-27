import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def coherence_pursuit(D0):
    '''
        N1 : The dimension of ambient space
        n1 : The number of inliers
        n2 : The number of outliers
        r : The rank of low rank matrix
    '''
    N1, N_sample = D0.shape

    # 1. Subtract the meidian
    D = D0 - np.median(D0, axis = 1, keepdims=True)

    # 2. Compute L2 norm of r-th columns of D0
    T = np.sum(D ** 2, axis = 0) ** 0.5 # rms amplitude
    T_aug = np.tile(T, (N1, 1)) # For element-wise division

    # 3. Normalize the columns of M0
    X = D/T_aug

    # 4. Compute the pairwise mutual coherence matrix
    G = np.matmul(X.T, X)

    # 5. Compute the coherence vector g as the L1-norm
    G = G - np.diag(np.diag(G))
    p = np.sum(G ** 2, axis = 0)
    p = p/np.max(p)


    # 7. Set # of samples by CoP algirithm to form the recovered subspace
    cut_off_th = 0.95
    n = sum(p >= cut_off_th)

    markerline, stemlines, baseline = plt.stem(p)
    plt.setp(markerline, marker='o', markersize=3, markeredgecolor="w", markeredgewidth=0.5)
    plt.setp(stemlines, linestyle="-", linewidth=0.1, color = 'white')
    plt.setp(baseline, linewidth=0.5)

    plt.plot([0, N_sample],  [cut_off_th, cut_off_th], 'r:')
    plt.title('The elements of vector p')
    plt.grid(True)
    plt.show()

    print(f'Threshold for cut-off: {cut_off_th}')
    print(f'# of samples by CoP algirithm to form the recovered subspace: {n}')
    
    b = np.argsort(p)
    b = b[::-1]
    Y = D[:, b[:n]]

    index_inliners = np.zeros(N_sample)
    index_inliners[b[:n]] = 1

    # Compute the factor by Singular Value Decomposition
    U, S, Vt = np.linalg.svd(Y.T, full_matrices = False)

    # flip eigenvectors' sign to enforce deterministic output
    U, Vt = svd_flip(U, Vt)

    components_ = Vt

    # Get variance explained by singular values
    explained_variance_ = (S**2) / (n - 1)
    singular_values_ = S.copy()  # Store the singular values.

    return T, components_, singular_values_, explained_variance_, index_inliners


def optimal_ncluster_BIC(p, n_range = range(1, 5)):
    clusterer_list, bic_list = [], []
    for k in n_range:
        clusterer = KMeans(n_clusters = k, n_init="auto", random_state=10, max_iter=300)
        clusterer.fit(p.reshape(-1, 1))
        clusterer_list.append(clusterer)
        
        cluster_labels = clusterer.predict(p.reshape(-1, 1))
        
        bic = 0
        n = p.shape[0]
        for ind in np.unique(cluster_labels):
            temp_p = p[cluster_labels == ind]
            ni = sum(cluster_labels == ind)
            term1 = ni * np.log(ni/n)
            term2 = ni/2 * np.log(2 * np.pi)
            term3 = ni/2 * np.log(np.var(temp_p))
            term4 = (ni - k)/2
            bic = bic + term1 - term2 - term3 - term4
        
        bic = bic - 1/2 * k * np.log(n)
        bic_list.append(bic)
    

    best_ind = np.argmax(bic_list)
    best_clusterer = clusterer_list[best_ind]

    if 1:
        plt.plot(n_range, bic_list)
        plt.plot(best_ind+1, bic_list[best_ind], 'ro')
        plt.show()

    # 6. Perform K-means clustering and select cut-off threshold (cut_off_th)
    best_clusterer = optimal_ncluster_BIC(p)

    cluster_labels = best_clusterer.predict(p.reshape(-1, 1))
    p_target = 0
    for ind in np.unique(cluster_labels):
        temp_p = p[cluster_labels == ind]
        if temp_p.mean() > p_target:
            ind_target = ind
            p_target = temp_p.mean()

    cut_off_th = p[cluster_labels == ind_target].min()

    return best_clusterer, cut_off_th

def match_sign_vector(y_true, y):
    for i in range(y_true.shape[0]):
        if y_true[i, 0] * y[i, 0] < 0:
            y[i, :] = - y[i, :]
    return y