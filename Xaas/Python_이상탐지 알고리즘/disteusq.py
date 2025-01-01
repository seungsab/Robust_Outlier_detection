import numpy as np

def disteusq(x, y, mode='0', w=None):
    
    nx, p = x.shape
    ny = y.shape[0]
    
    if mode == '0':
        mode = 'd' if nx == ny else 'x'
    
    if mode == 'd' or (mode != 'x' and nx == ny):
        nx = min(nx, ny)
        z = x[:nx, :] - y[:nx, :]
        if w is None:
            d = np.sum(np.conj(z) * z, axis=1)
        elif len(w.shape) == 1:
            wv = w.flatten()
            d = np.sum(np.conj(z) * z * wv[:, np.newaxis], axis=1)
        else:
            d = np.sum(np.dot(z, w) * np.conj(z), axis=1)
    else:
        if p > 1:
            if w is None:
             
                z = np.transpose(np.expand_dims(x, axis=2).repeat(ny, axis=2), (0, 2, 1)) - np.transpose(np.expand_dims(y, axis=2).repeat(nx, axis=2), (2, 0, 1))
               
                d = np.sum(np.conj(z) * z, axis=2)
            else:
                nxy = nx * ny
                z = np.transpose(np.expand_dims(x, axis=2).repeat(ny, axis=2), (0, 2, 1)) - (np.transpose(np.expand_dims(y, axis=2).repeat(nx, axis=2), (2, 0, 1)), nxy, p)
                if len(w.shape) == 1:
                    wv = w.flatten()
                    d = np.sum(np.conj(z) * z * wv[:, np.newaxis], axis=1).reshape(nx, ny)
                else:
                    d = np.sum(np.dot(z, w) * np.conj(z), axis=1).reshape(nx, ny)
        else:
            z = x[:, np.newaxis] - y[:, np.newaxis].T
            if w is None:
                d = np.conj(z) * z
            else:
                d = w * np.conj(z) * z
    
    if 's' in mode:
        d = np.sqrt(d)
    
    return d
