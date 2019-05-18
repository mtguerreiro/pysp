import numpy as np
import scipy.signal


def tf_from_poles(poles):
    """Computes the numerator and denominator of the filter's transfer
    function based on its poles.

    Parameters
    ----------
    poles : np.ndarray
        Array with poles. 
    Returns
    -------
    num : np.ndarray
        Array with numerator coefficients.

    den : np.ndarray
        Array with denominator coefficients.

    """
    den = np.poly(poles).real
    num = np.array([den[-1]])

    return num, den    

    
def tf_to_parallel_sos(num, den):
    """Expands a transfer function as a sum of second-order polynomials.

    Parameters
    ----------
    num : np.ndarray
        Transfer function numerator coefficients.

    den : np.ndarray
        Transfer function denominator coefficients.

    Returns
    -------
    num : np.ndarray
        Array with numerator coefficients for each 2nd order term.

    den : np.ndarray
        Array with denonimator coefficients for each 2nd order term.

    """
    Ak, pk, k = scipy.signal.residue(num, den)
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
            den_1 = np.poly(pk[k])
            den_2 = np.poly(pk[k+1])
            den = np.polymul(den_1, den_2)
            num = Ak[k]*den_2 + Ak[k+1]*den_1

            den_list.append(den.real)
            num_list.append(num.real)

            # Skips next pole since we took one and its conjugate
            k+=1
        else:
            den_list.append(np.poly(pk[k]).real)
            num_list.append(np.poly(Ak[k]).real)
            
        k+=1

    return np.array(num_list), np.array(den_list)  


def tf_to_parallel_sos_z(num, den, T):
    """Builds a discrete-model as a sum of second-order polynomials from a
    continuous-time model.

    Parameters
    ----------
    num : np.ndarray
        Numerator coefficients of the continuous-time model.

    den : np.ndarray
        Denominator coefficients of the continuous-time model.
        
    T : int, float
        Sampling period

    Returns
    -------
    num : np.ndarray
        Array with numerator coefficients for each 2nd order term.

    den : np.ndarray
        Array with denonimator coefficients for each 2nd order term.
        
    """    
    Ak, pk, k = scipy.signal.residue(num, den)
    print(Ak, pk)
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


def parallel_sos_to_tf(num, den):
    """Builds a transfer function from the SOS representation.

    Parameters
    ----------
    num : np.ndarray
        Array with denominator coefficients for each section.

    den : np.ndarray
        Array with numerator coefficients for each section.

    Returns
    -------
    num : np.ndarray
        Array with numerator for the transfer function.

    den : np.ndarray
        Array with denominator for the transfer function.
    
    """
    _num = np.array(num[0])
    _den = np.array(den[0])

    for n in range(1, num.shape[0]):
        _num = np.polyadd(np.polymul(num[n], _den), np.polymul(_num, den[n]))
        _den = np.polymul(den[n], _den)

    return _num, _den


def sos_filter(sos, x):
    """Filters a signal considering a single SOS section

    Parameters
    ----------
    sos : tuple, list, np.ndarray
        Numerator and denominator of the section's transfer function.

    x : np.ndarray
        1-D vector with filter's input.

    Returns
    -------
    y : np.ndarray
        Filter's output
        
    """
    N = x.shape[0]
    num = sos[0]
    den = sos[1]

    if num.shape[0] != 3:
        n = num.shape[0]
        num = np.hstack((num, np.zeros(3 - n)))
    if den.shape[0] != 3:
        n = den.shape[0]
        den = np.hstack((den, np.zeros(3 - n)))

    den = den[::-1]
    num = num[::-1]

    y = np.zeros(x.shape)
    y[0] = num[0]*x[0]
    y[1] = -den[1]*y[0] + num[-1]*x[1] + num[-2]*x[0]

    for n in range(2, N):
        y[n] = y[(n - 2):n] @ -den[:-1] + x[(n - 2):(n + 1)] @ num

    return y


def sos_filter_rt(sos, x):
    """Filters a signal considering a single SOS section

    Parameters
    ----------
    sos : tuple, list, np.ndarray
        Numerator and denominator of the section's transfer function.

    x : np.ndarray
        1-D vector with filter's input.

    Returns
    -------
    y : np.ndarray
        Filter's output
        
    """
    N = x.shape[0]
    num = sos[0]
    den = sos[1]

    if num.shape[0] != 3:
        n = num.shape[0]
        num = np.hstack((num, np.zeros(3 - n)))
    if den.shape[0] != 3:
        n = den.shape[0]
        den = np.hstack((den, np.zeros(3 - n)))

    y = np.zeros(x.shape)
    
    v = num[2]*x[0]
    u = num[1]*x[0]
    y[0] = num[0]*x[0]
    
    for n in range(1, N):
        y[n] = num[0]*x[n] + u
        u = num[1]*x[n] + -den[1]*y[n] + v
        v = num[2]*x[n] + -den[2]*y[n]
        
    return y


def filter_fir(fir, x):

    num = fir[0][::-1]

    N = x.shape[0]
    M = num.shape[0] - 1

    # Append initial zeros to fix number of input samples
    xt = np.hstack((np.zeros(M), x))

    y = np.zeros(xt.shape)

    for n in range(M, N + M):
        y[n] = xt[(n - M):(n + 1)] @ num

    return y[M:]

    
