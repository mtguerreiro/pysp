import numpy as np

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
