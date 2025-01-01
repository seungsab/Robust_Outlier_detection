import numpy as np


def standardization_data(X0, mu0, sds0, numobs, SD_type):
    if SD_type == 'Mean_centered':
        X0std = X0 - np.tile(mu0, (numobs, 1))
    elif SD_type == 'Z-score':
        X0std = (X0 - np.tile(mu0, (numobs, 1))) / np.tile(sds0, (numobs, 1))
    return X0std