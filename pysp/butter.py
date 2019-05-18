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

    
##def tf_to_sos(num, den):
##    """Builds a 2nd order model for a Butterworth filter with the given poles.
##
##    Parameters
##    ----------
##    poles : np.ndarray
##        Array with filter poles.
##
##    Returns
##    -------
##    num, den : np.ndarray
##        The numerator and denominator for the filter's transfer function as a
##        sum of second order terms.
##    """
##
##    Ak, pk, k = scipy.signal.residue(num, den)
##    pk_sort = np.argsort(pk)
##    Ak = Ak[pk_sort]
##    pk = pk[pk_sort]
##
##    den_list = []
##    num_list = []
##    k = 0    
##    while k < len(Ak):
##        # Check if next pole is complex. If not, just adds partial numerator
##        # and denominator to list
##        if np.abs(np.imag(pk[k])) > 1e-10:
##            # Forms 2nd order partial transfer function by combining one
##            # complex pole with its conjugate
##            den_1 = np.poly(pk[k])
##            den_2 = np.poly(pk[k+1])
##            den = np.polymul(den_1, den_2)
##            num = Ak[k]*den_2 + Ak[k+1]*den_1
##
##            den_list.append(den.real)
##            num_list.append(num.real)
##
##            # Skips next pole since we took one and its conjugate
##            k+=1
##        else:
##            den_list.append(np.poly(pk[k]).real)
##            num_list.append(np.poly(Ak[k]).real)
##            
##        k+=1
##
##    return np.array(num_list), np.array(den_list)  
##
##
##def tf_to_sos_z(num, den, T):
##    """Builds a 2nd order discrete model for a Butterworth filter with
##    the given poles and sampling period.
##
##    Parameters
##    ----------
##    poles : np.ndarray
##        Array with filter poles.
##        
##    T : int, float
##        Sampling period
##
##    Returns
##    -------
##    num, den : np.ndarray
##        The numerator and denominator for the filter's transfer function as a
##        sum of second order terms.
##        
##    """    
##    Ak, pk, k = scipy.signal.residue(num, den)
##    print(Ak, pk)
##    pk_sort = np.argsort(pk)
##    Ak = Ak[pk_sort]
##    pk = pk[pk_sort]
##
##    den_list = []
##    num_list = []
##    k = 0    
##    while k < len(Ak):
##        # Check if next pole is complex. If not, just adds partial numerator
##        # and denominator to list
##        if np.abs(np.imag(pk[k])) > 1e-10:
##            # Forms 2nd order partial transfer function by combining one
##            # complex pole with its conjugate
##            den_1 = np.poly(1/np.exp(T*-pk[k]))
##            den_2 = np.poly(1/np.exp(T*-pk[k+1]))
##            den = np.polymul(den_1, den_2)
##            num = T*Ak[k]*den_2 + T*Ak[k+1]*den_1
##
##            den_list.append(den.real)
##            num_list.append(num.real)
##
##            # Skips next pole since we took one and its conjugate
##            k+=1
##        else:
##            den_list.append(np.poly(1/np.exp(T*-pk[k])).real)
##            num_list.append(np.poly(T*Ak[k]).real)
##            
##        k+=1
##
##    return np.array(num_list), np.array(den_list)
##
##
##def sos_to_tf(num, den):
##    """Builds a transfer function from the SOS representation.
##
##    Parameters
##    ----------
##    num : np.ndarray
##        Array with denominator coefficients for each section.
##
##    den : np.ndarray
##        Array with numerator coefficients for each section.
##
##    Returns
##    -------
##    num : np.ndarray
##        Array with numerator for the transfer function.
##
##    den : np.ndarray
##        Array with denominator for the transfer function.
##    
##    """
##    _num = np.array(num[0])
##    _den = np.array(den[0])
##
##    for n in range(1, num.shape[0]):
##        _num = np.polyadd(np.polymul(num[n], _den), np.polymul(_num, den[n]))
##        _den = np.polymul(den[n], _den)
##
##    return _num, _den
##
##
##def sos_filter(sos, x):
##    """Filters a signal considering a single SOS section
##
##    Parameters
##    ----------
##    sos : tuple, list, np.ndarray
##        Numerator and denominator of the section's transfer function.
##
##    x : np.ndarray
##        1-D vector with filter's input.
##
##    Returns
##    -------
##    y : np.ndarray
##        Filter's output
##        
##    """
##    N = x.shape[0]
##    num = sos[0]
##    den = sos[1]
##
##    if num.shape[0] != 3:
##        n = num.shape[0]
##        num = np.hstack((num, np.zeros(3 - n)))
##    if den.shape[0] != 3:
##        n = den.shape[0]
##        den = np.hstack((den, np.zeros(3 - n)))
##
##    den = den[::-1]
##    num = num[::-1]
##
##    y = np.zeros(x.shape)
##    y[0] = num[0]*x[0]
##    y[1] = -den[1]*y[0] + num[-1]*x[1] + num[-2]*x[0]
##
##    for n in range(2, N):
##        y[n] = y[(n - 2):n] @ -den[:-1] + x[(n - 2):(n + 1)] @ num
##
##    return y
##
##
##def sos_filter_rt(sos, x):
##    """Filters a signal considering a single SOS section
##
##    Parameters
##    ----------
##    sos : tuple, list, np.ndarray
##        Numerator and denominator of the section's transfer function.
##
##    x : np.ndarray
##        1-D vector with filter's input.
##
##    Returns
##    -------
##    y : np.ndarray
##        Filter's output
##        
##    """
##    N = x.shape[0]
##    num = sos[0]
##    den = sos[1]
##
##    if num.shape[0] != 3:
##        n = num.shape[0]
##        num = np.hstack((num, np.zeros(3 - n)))
##    if den.shape[0] != 3:
##        n = den.shape[0]
##        den = np.hstack((den, np.zeros(3 - n)))
##
##    y = np.zeros(x.shape)
##    
##    v = num[2]*x[0]
##    u = num[1]*x[0]
##    y[0] = num[0]*x[0]
##    
##    for n in range(1, N):
##        y[n] = num[0]*x[n] + u
##        u = num[1]*x[n] + -den[1]*y[n] + v
##        v = num[2]*x[n] + -den[2]*y[n]
##        
##    return y
##
##
##def filter_fir(fir, x):
##
##    num = fir[0][::-1]
##
##    N = x.shape[0]
##    M = num.shape[0] - 1
##
##    # Append initial zeros to fix number of input samples
##    xt = np.hstack((np.zeros(M), x))
##
##    y = np.zeros(xt.shape)
##
##    for n in range(M, N + M):
##        y[n] = xt[(n - M):(n + 1)] @ num
##
##    return y[M:]

    
