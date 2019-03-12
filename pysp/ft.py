import numpy as np
import pysp.utils as sputils
import time

def dft_naive(x):
    r"""Computes the naive DFT of a sequence.

    The DFT is computed as defined by:

    .. math::
        :label: eq-naive-dft

        X_k = \sum^{N-1}_{n=0}x_n e^{-j2\pi kn/N}

    where :math:`X_k` is  the :math:`k`th Fourier coefficients, :math:`x_n`
    is the :math:`n`th sample of the input signal and :math:`N` is the size
    of the input signal :math:`x`. 

    Parameters
    ----------
    x : np.ndarray
        Input sequence.

    Returns
    -------
    np.ndarray:
        Fourier coefficients of the input signal.
        
    """
    N = x.shape[0]

    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)

    cos = np.cos(2*np.pi*k*n/N)
    sin = np.sin(2*np.pi*k*n/N)
    
    X = (cos - 1j*sin) @ x

    return X


def idft_naive(X):
    r"""Computes the naive inverse DFT of a sequence.

    The inverse DFT is computed as defined by:

    .. math::
        :label: eq-naive-idft

        x_n = \frac{1}{N}\sum^{N-1}_{k=0}X_k e^{j2\pi kn/N}

    where :math:`X_k` is the :math:`k`th Fourier coefficient of the input
    signal, :math:`x_n` is the :math:`n`th sample of the inversed signal and
    :math:`N` is the size of the input sequence :math:`X`.

    Parameters
    ----------
    X : np.ndarray
        Input sequence.

    Returns
    -------
    np.ndarray:
        Inverse DFT of the input signal.
        
    """
    N = X.shape[0]

    n = np.arange(N).reshape(-1, 1)
    k = np.arange(N)

    cos = np.cos(2*np.pi*k*n/N)
    sin = np.sin(2*np.pi*k*n/N)

    x = (cos + 1j*sin) @ X

    return x/N


def fft(x):
    r"""Computes the DFT of a sequence using the FFT algorithm.

    Parameters
    ----------
    x : np.ndarray
        Input sequence.

    Returns
    -------
    np.ndarray:
        Fourier coefficients of the input signal.
        
    """
    # Signal length and number of stages
    N = int(x.shape[0])
    n = int(np.log10(N)/np.log10(2))

    # FFT indexes
    idx = sputils.samples_bit_reversal(np.arange(N))
    
    # Rearanges samples
    X = np.zeros(x.shape, complex)
    X[:] = x[idx]

    for m in range(n):
        # Number of elements in each block
        nb = 2**m
        # Number of blocks
        mb = N/2**(m+1)

        alp = np.array([np.arange(nb) + 2*nb*k for k in range(int(mb))]).flatten()
        bet = np.array([np.arange(nb) + 2*nb*k + nb for k in range(int(mb))]).flatten()

        _k = 2**(n - m - 1)*np.arange(N)
        Wnk = np.exp(-1j*2*np.pi*_k/N)

        a = sputils.butterfly(X[alp], X[bet], Wnk[alp])
        X[alp] = a[0]
        X[bet] = a[1]

    return X    
