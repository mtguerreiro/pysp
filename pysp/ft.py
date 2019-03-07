import numpy as np

def dft_naive(x):

    N = x.shape[0]
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)

    cos = np.cos(2*np.pi*k*n/N)
    sin = np.sin(2*np.pi*k*n/N)
    
    X = x*(cos - 1j*sin)

    return np.sum(X, 1)
