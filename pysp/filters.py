import numpy as np
import scipy.signal

def butter(wp, Hwp, ws, Hws, cutoff='wp', T=1, show_partial=False):
    """Determines the order and cut-off frequency for a Butterworth filter,
    based on the specifications of pass-band/stop-band frequency/attenuation.

    Continue with docs...

    Parameters
    ----------
    wp : int, float
        Pass-band frequency.
        
    Hwp : int, float
        Squared magnitude of :math:`|H(j\omega)|` desided at the pass-band
        frequency.

    ws : int, float
        Stop-band frequency.

    Hws : int, float
        Squared magnitude of :math:`|H(j\omega)|` desided at the stop-band
        frequency.
        
    """
    A = lambda H_w : 1/H_w**2 - 1

    # Corner frequency
    _wc = butter_corner(wp, Hwp, ws, Hws, cutoff)

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

    butter_tf(sk, T)

    butter_filter = {}
    butter_filter['wc'] = wc
    butter_filter['N'] = N
    butter_filter['poles'] = sk

    # Printing
    if show_partial is True:
        print('Order: ', _N)
        print('Cut-off: ', _wc)
        
        print('Poles:')
        print(sk)

    return butter_filter


def butter_corner(wp, Hwp, ws, Hws, cutoff='wp'):

    A = lambda H_w : 1/H_w**2 - 1
    
    log_wc = np.log10(A(Hwp))*np.log10(ws) - np.log10(A(Hws))*np.log10(wp)
    log_wc = log_wc/(np.log10(A(Hwp)/A(Hws)))

    return 10**log_wc


def butter_order(wc, w, Hw):

    A = lambda H_w : 1/H_w**2 - 1

    return np.log10(A(Hw))/(2*np.log10(w/wc))


def butter_poles(wc, N):

    k = np.arange(2*N)
    return wc*np.exp((1j*np.pi/2/N)*(2*k + N - 1))


def butter_tf(poles, T):

    sk_poly = np.poly(poles[poles.real < 0])
    
    Ak, pk, k = scipy.signal.residue(sk_poly[-1], sk_poly)
    pk_sort_idx = np.argsort(pk)
    Ak = Ak[pk_sort_idx]
    pk = pk[pk_sort_idx]

    k = 0
    while k < len(Ak):

        if np.abs(np.imag(pk[k])) > 1e-10:
            a1 = np.poly(1/np.exp(T*-pk[k]))
            a2 = np.poly(1/np.exp(T*-pk[k+1]))
            a = np.polymul(a1, a2)
            A = T*Ak[k]*a2 + T*Ak[k+1]*a1

            print('Den: ')
            print(a.real)
            print('Num: ')
            print(A.real)

            k+=1

        k+=1
