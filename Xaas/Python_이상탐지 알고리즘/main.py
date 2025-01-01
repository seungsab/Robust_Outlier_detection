import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.decomposition import PCA
from scipy.stats import chi2
from tqdm import tqdm

from standardization_data import standardization_data
from select_retained_nPC import select_retained_nPC
from vmwpca_update import VMWPCA_update


# option1
# data = pd.read_csv('Data/Z24.csv')

# option2
data = pd.read_csv('Data/CG_1&CG_2&TT_1_sig0_3_12.csv')
data = data.iloc[::10, :].reset_index(drop=True)

time_stamp = pd.to_datetime(data.iloc[:, 0])
f = data.iloc[:, 1:5].values
T_avg = data.iloc[:, 5].values
init_train_ratio = 0.7

# option1
# indx2 = int(time_stamp[time_stamp == datetime(1998, 8, 9)].index[0])
# option2
indx2 = int(time_stamp[time_stamp > datetime(2023, 10, 1)].index[0])

indx1 = int(indx2 * init_train_ratio)

# option1
# measurement_type = 'dynamic'
# option2
measurement_type = 'static'

Variable_window = 1
PCA_par = {
    'SD_type': 'Z-score', 
    'alpha': '99%', 
    'nPC_select_algorithm': 'eigengap',
    'Monitoring_statistic': [1, 0],
}
if measurement_type == 'static':
    PCA_par['n_stall'] = 120
    L_min=3600
    L_max=14400
else:
    PCA_par['n_stall'] = 10
    L_min=500
    L_max=2000

PCA_par['x'] = [[0, indx1], [indx1+1, indx2], [indx2+1, len(f) -1]]
PCA_par['d_indx'] = indx2

zerIdx = []
for i in range(len(T_avg)-1):
    if (T_avg[i] > 0 and T_avg[i+1] < 0) or (T_avg[i] < 0 and T_avg[i+1] > 0):
        zerIdx.append(i)

x0 = np.arange(PCA_par['x'][0][0], PCA_par['x'][0][1]+1)
x1 = np.arange(PCA_par['x'][1][0], PCA_par['x'][1][1]+1)
x2 = np.arange(PCA_par['x'][2][0], PCA_par['x'][2][1]+1)

ts1 = pd.DataFrame(data.iloc[:, 1:7].values, columns=[f'Freq_{i}' for i in range(1, 7)], index=pd.to_datetime(time_stamp))


plt.figure(figsize=(10, 6))


plt.subplot(4, 1, 1)
plt.plot(time_stamp[x0], f[x0, :], 'bo', markersize=2, label='Training Data')
plt.plot(time_stamp[x1], f[x1, :], 'g^', markersize=2, label='Validation Data')
plt.plot(time_stamp[x2], f[x2, :], 'kx', markersize=3, label='Testing Data')
plt.ylabel('Frequency (Hz)', fontsize=15)
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_stamp[x0], T_avg[x0], 'b.', markersize=2, label='Training Data')
plt.plot(time_stamp[x1], T_avg[x1], 'g.', markersize=2, label='Validation Data')
plt.plot(time_stamp[x2], T_avg[x2], 'k.', markersize=3, label='Testing Data')
plt.ylabel('Temperature (°C)', fontsize=15)
plt.grid(True)

plt.tight_layout()
plt.show()

PCA_par['T_avg'] = T_avg

SD_type = PCA_par['SD_type']
alpha = PCA_par['alpha']
nPC_select_algorithm = PCA_par['nPC_select_algorithm']
Monitoring_statistic = PCA_par['Monitoring_statistic']

PC = {}
PC['n_stall'] = PCA_par['n_stall']
PC['Monitoring_statistic'] = PCA_par['Monitoring_statistic']
PC['k_cluster'] = list(range(1, 5))

numobs_f, numvars_f = f.shape



X0 = f[x0, :]
numobs, numvars = X0.shape
mu0 = np.mean(X0, axis=0)
sds0 = np.std(X0, axis=0)
X0std = standardization_data(X0, mu0, sds0, numobs, SD_type)

pca = PCA()
pca.fit(X0std)
loadings = pca.components_.T
scores = pca.transform(X0std)
variances = pca.explained_variance_

n_pc = select_retained_nPC(variances, nPC_select_algorithm)

reconstructed_X0 = np.dot(scores[:, :n_pc[-1]], loadings[:, :n_pc[-1]].T)
residuals = X0std - reconstructed_X0
Qdist = np.sqrt(np.sum(residuals**2, axis=1))

m_Q = np.mean(Qdist)
V_Q = np.var(Qdist)

V = 2 * (m_Q**2) / V_Q
distcrit = (V_Q / (2 * m_Q)) * np.array([chi2.ppf(0.95, V), chi2.ppf(0.99, V)])

print("Critical limits (95% and 99%):", distcrit)





Time_effors = []
tstart = time.time()
telapsed = time.time() - tstart
Time_effors.append(telapsed)


print("Elapsed time for Initial_learning_PCA:", telapsed)

X1 = {}
X1['f0'] = X0
X1['mu0'] = np.tile(mu0, (Qdist.shape[0], 1)) 
X1['sds0'] = np.tile(sds0, (Qdist.shape[0], 1))


PC['loadings'] = loadings
PC['scores'] = scores
PC['n_pc'] = n_pc
PC['variances'] = variances
PC['update'] = 1

Q = {}
Q['Qdist'] = Qdist
Q['m_Q'] = m_Q
Q['V_Q'] = V_Q
Q['distcrit'] = np.tile(distcrit, (Qdist.shape[0], 1))

if Variable_window:
    X1['L_min'] = L_min
    X1['L_max'] = L_max
    X1['L'] = []
    X1['N_cluster'] = []
    X1['BIC'] = []

    if Qdist.shape[0] < X1['L_max']:
        X1['L'] = np.tile(Qdist.shape[0], (Qdist.shape[0], 1)).flatten()
        X1['L0'] = X1['L']
    else:
        X1['L'] = np.tile(X1['L_max'], (Qdist.shape[0], 1)).flatten()
        X1['L0'] = X1['L']
else:
    X1['f'] = X0


X1['T_avg'] = PCA_par['T_avg']
X1['ts1'] = ts1

Q['distcrit'] = np.vstack([Q['distcrit'], Q['distcrit'][-1, :]])

PC['fault'] = []
PC['outlier'] = []

if alpha == '95%':
    Q_lim = Q['distcrit'][-1, 0]
else:
    Q_lim = Q['distcrit'][-1, 1]

PC['outlier'] = np.where(Q['Qdist'] > Q_lim)[0]

for i in tqdm(range(0, len(x1)), desc="Please wait...Validation samples"):
    X1['f'] = np.vstack([X1['f0'][-X1['L'][-1]:, :], f[x1[i], :]])

    X1, PC, Q = VMWPCA_update(X1, PC, Q, alpha, SD_type, nPC_select_algorithm, x1[0] + i)




for i in tqdm(range(0, len(x2)), desc="Testing samples", unit="sample"):
    X1['f'] = np.vstack([X1['f0'][-X1['L'][-1]:, :], f[x2[0] + i, :]])

    X1, PC, Q = VMWPCA_update(X1, PC, Q, alpha, SD_type, nPC_select_algorithm, x2[0] + i)



print("Testing completed!")
fn_fig = f'VMWPCA n={x0[-1]}'
fig_T_Q = plt.figure()

plt.figure(figsize=(8, 4))
plt.plot(ts1.index[x0], Q['Qdist'][x0], 'bo', markersize=3, markerfacecolor='b', label='x0')
plt.plot(ts1.index[x1], Q['Qdist'][x1], 'v', markerfacecolor=(0, 0.498, 0), markeredgecolor=(0, 0.498, 0), markersize=3, label='x1')
plt.plot(ts1.index[x2], Q['Qdist'][x2], 'kx', markersize=3, label='x2')
plt.plot(ts1.index[PC['outlier']], Q['Qdist'][PC['outlier']], 'rs', markersize=3, label='Outliers')

if alpha == '95%':
    plt.plot(ts1.index, Q['distcrit'][:-1, 0], 'k-.', markersize=3, linewidth=1.5, label='95% Threshold')
elif alpha == '99%':
    plt.plot(ts1.index, Q['distcrit'][:-1, 1], 'k-.', markersize=3, linewidth=1.5, label='99% Threshold')

plt.ylabel('Q-statistics (SPE)', fontsize=15, fontweight='bold')
plt.xlabel('Time', fontsize=15, fontweight='bold')
plt.grid(True)
plt.legend()

YLIM = plt.gca().get_ylim()
plt.fill_between([ts1.index[PCA_par['d_indx']], ts1.index[-1]], YLIM[0], YLIM[1], color='yellow', alpha=0.1, edgecolor='k', label='Area')

plt.title(f'{fn_fig}, Alpha={alpha}', fontsize=15, fontweight='bold')

plt.show()