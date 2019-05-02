import numpy as np
import scipy.signal


def tf_to_sos(num, den):
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


def tf_to_sos_z(num, den, T):
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


def sos_to_tf(num, den):

    _num = np.array([1.])
    _den = np.array([1.])

    for n in range(num.shape[0]):
        _num = np.polyadd(np.polymul(num[n], _den), np.polymul(_num, den[n]))
        _den = np.polymul(den[n], _den)

    return _num, _den
    
