import numpy as np
import scipy.signal

def butter(wp, Hwp, ws, Hws, cutoff='wp', T=1, show_partial=False):
    r"""Designs a low-pass Butterworth filter based on the given parameters.

    Continue with docs...

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
        :math:`\omega_p`.

    T : int, float
        Sampling time to design the discrete filter.
        
    """
    A = lambda H_w : 1/H_w**2 - 1

    # Corner frequency
    _wc = butter_corner(wp, Hwp, ws, Hws)

    # Order
    _N = butter_order(_wc, wp, Hwp)
    
    # Actual order, cut-off
    N = np.ceil(_N).astype(int)
    if cutoff == 'wp':
        wc = wp/(A(Hwp)**(1/2/N))
    else:
        wc = ws/(A(Hws)**(1/2/N))
        
    # Poles
    sk = butter_poles(wc, N)

    # Continuous TF
    numc, denc = butter_tf(sk)

    # Discrete TF
    numd, dend = butter_tf_discrete(sk, T)

    # Builds filter dictionary
    butter_filter = {}
    butter_filter['wc'] = wc
    butter_filter['N'] = N
    butter_filter['poles'] = sk
    butter_filter['Hs'] = [numc, denc]
    butter_filter['Hz'] = [numd, dend]

    # Infos
    if show_partial is True:
        print('Exact order: ', _N)
        print('Exact cut-off: ', _wc)
        
        print('\nPoles:')
        print(sk[sk.real < 0])

    return butter_filter


def butter_corner(wp, Hwp, ws, Hws):
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


def butter_order(wc, w, Hw):
    r"""Computer the order necessary for a Butterworth filter with the
    specified cut-off frequency and desired magnitude :math:`H(j\omega)`
    at frequency :math:`\omega`.

    Parameters
    ----------
    wc : int, float
        Cut-off frequency :math:`\omega_c`.

    w : int, float
        Frequency :math:`\omega` of the specified desired magnitude.
        
    Hwp : int, float
        Magnitude of :math:`H(j\omega)` desired at the specified frequency
        :math:`\omega`.

    Returns
    -------
    N : float
        Exact order necessary.
        
    """
    A = lambda H_w : 1/H_w**2 - 1

    return np.log10(A(Hw))/(2*np.log10(w/wc))


def butter_poles(wc, N):
    """Computes the poles of a Butterworth filter with the specified cut-off
    frequency :math:`\omega_c` and order :math:`N`. 

    Parameters
    ----------
    wc : int, float
        Cut-off frequency :math:`\omega_c`.

    N : int, float
        Filter order :math:`N`.

    Returns
    -------
    sk : np.ndarray
        Array with poles.
        
    """
    k = np.arange(2*N)
    return wc*np.exp((1j*np.pi/2/N)*(2*k + N - 1))


def butter_tf(poles):
    """Builds a 2nd order model for a Butterworth filter with the given poles.

    Parameters
    ----------
    poles : np.ndarray
        Array with filter poles.

    Returns
    -------
    num, den : np.ndarray
        The numerator and denominator for the filter's transfer function as a
        sum of second order terms.
    """
    sk_poly = np.poly(poles[poles.real < 0])

    Ak, pk, k = scipy.signal.residue(sk_poly[-1], sk_poly)
    pk_sort = np.argsort(pk)
    Ak = Ak[pk_sort]
    pk = pk[pk_sort]

    den_list = []
    num_list = []
    k = 0    
    while k < len(Ak):
        # Check if next pole is complex. If not, just adds partial numerator
        # and denominator to list
        if np.abs(np.imag(pk[k])) > 1e-10:
            # Forms 2nd order partial transfer function by combining one
            # complex pole with its conjugate
            den_1 = np.poly(-pk[k])
            den_2 = np.poly(-pk[k+1])
            den = np.polymul(den_1, den_2)
            num = Ak[k]*den_2 + Ak[k+1]*den_1

            den_list.append(den.real)
            num_list.append(num.real)

            # Skips next pole since we took one and its conjugate
            k+=1
        else:
            den_list.append(np.poly(-pk[k]).real)
            num_list.append(np.poly(Ak[k]).real)
            
        k+=1

    return np.array(num_list), np.array(den_list)    

def butter_tf_discrete(poles, T):
    """Builds a 2nd order discrete model for a Butterworth filter with
    the given poles and sampling period.

    Parameters
    ----------
    poles : np.ndarray
        Array with filter poles.
        
    T : int, float
        Sampling period

    Returns
    -------
    num, den : np.ndarray
        The numerator and denominator for the filter's transfer function as a
        sum of second order terms.
        
    """
    sk_poly = np.poly(poles[poles.real < 0])
    
    Ak, pk, k = scipy.signal.residue(sk_poly[-1], sk_poly)
    pk_sort = np.argsort(pk)
    Ak = Ak[pk_sort]
    pk = pk[pk_sort]

    den_list = []
    num_list = []
    k = 0    
    while k < len(Ak):
        # Check if next pole is complex. If not, just adds partial numerator
        # and denominator to list
        if np.abs(np.imag(pk[k])) > 1e-10:
            # Forms 2nd order partial transfer function by combining one
            # complex pole with its conjugate
            den_1 = np.poly(1/np.exp(T*-pk[k]))
            den_2 = np.poly(1/np.exp(T*-pk[k+1]))
            den = np.polymul(den_1, den_2)
            num = T*Ak[k]*den_2 + T*Ak[k+1]*den_1

            den_list.append(den.real)
            num_list.append(num.real)

            # Skips next pole since we took one and its conjugate
            k+=1
        else:
            den_list.append(np.poly(1/np.exp(T*-pk[k])).real)
            num_list.append(np.poly(T*Ak[k]).real)
            
        k+=1

    return np.array(num_list), np.array(den_list)
