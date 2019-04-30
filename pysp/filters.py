import numpy as np


def butter(wp, Hwp, ws, Hws, cutoff='wp', show_partial=False):
    """Determines the order and cut-off frequency for a Butterworth filter,
    based on the specifications of pass-band/stop-band frequency/attenuation.

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
    A = lambda Hw : 1/Hw**2 - 1

    # Corner frequency
    log_wc = np.log10(A(Hwp))*np.log10(ws) - np.log10(A(Hws))*np.log10(wp)
    log_wc = log_wc/(np.log10(A(Hwp)/A(Hws)))
    _wc = 10**log_wc

    # Order
    _N = np.log10(A(Hwp))/(2*np.log10(wp/_wc))
    
    # Actual order, cut-off
    N = np.ceil(_N).astype(int)
    if cutoff == 'wp':
        wc = wp/(A(Hwp)**(1/2/N))
    else:
        wc = ws/(A(Hws)**(1/2/N))

    if show_partial is True:
        print('Order: ', _N)
        print('Cut-off: ', _wc)
        
    return wc, N
