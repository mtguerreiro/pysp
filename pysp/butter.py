import numpy as np
import scipy.signal


def corner(wp, Hwp, ws, Hws):
    r"""Computes the corner frequency necessary for a Butterworth low-pass
    filter with the given specifications.

    Parameters
    ----------
    wp : int, float
        Pass-band frequency :math:`\omega_p`.
    
    Hwp : int, float
        Magnitude of :math:`H(j\omega)` desided at the pass-band frequency
        :math:`\omega_p`.

    ws : int, float
        Stop-band frequency :math:`\omega_s`.

    Hws : int, float
        Magnitude of :math:`H(j\omega)` desided at the stop-band frequency
        :math:`\omega_s`.

    Returns
    -------
    wc : float
        Cut-off frequency :math:`\omega_c`.
            
    """
    A = lambda H_w : 1/H_w**2 - 1
    log_wc = np.log10(A(Hwp))*np.log10(ws) - np.log10(A(Hws))*np.log10(wp)
    log_wc = log_wc/(np.log10(A(Hwp)/A(Hws)))

    return 10**log_wc


def order(wc, w, Hw):
    r"""Computer the order necessary for a Butterworth filter with the
    specified cut-off frequency and desired magnitude :math:`H(j\omega)`
    at frequency :math:`\omega`.

    Parameters
    ----------
    wc : int, float
        Cut-off frequency :math:`\omega_c`.

    w : int, float
        Frequency :math:`\omega` of the specified desired magnitude.
            
    Hw : int, float
        Magnitude of :math:`H(j\omega)` desired at the specified frequency
        :math:`\omega`.

    Returns
    -------
    N : float
        Exact order necessary.
            
    """
    A = lambda H_w : 1/H_w**2 - 1
    
    return np.log10(A(Hw))/(2*np.log10(w/wc))


def poles(N, wc):
    """Computes the poles of a Butterworth filter with the specified
    cut-off frequency :math:`\omega_c` and order :math:`N`. 

    Parameters
    ----------
    N : int, float
        Filter order :math:`N`.

    wc : int, float
        Cut-off frequency :math:`\omega_c`.

    Returns
    -------
    sk : np.ndarray
        Array with poles.
            
    """
    k = np.arange(2*N)

    return wc*np.exp((1j*np.pi/2/N)*(2*k + N - 1))
