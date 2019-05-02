import numpy as np
import scipy.signal
import pysp.filter_utils as futils


class butter:
    r"""Creates a low-pass Butterworth filter based on the given parameters.

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
    def __init__(self, wp, Hwp, ws, Hws, cutoff='wp', T=1, show_partial=False):
        self._A = lambda H_w : 1/H_w**2 - 1

        # Sampling time for discrete stuff
        self.T = T

        # Corner frequency
        _wc = self.__corner(wp, Hwp, ws, Hws)
        # Order
        _N = self.__order(_wc, wp, Hwp)
        
        # Actual order, cut-off
        N = np.ceil(_N).astype(int)
        if cutoff == 'wp':
            wc = wp/(self._A(Hwp)**(1/2/N))
        else:
            wc = ws/(self._A(Hws)**(1/2/N))
        self.order = N
        self.wc = wc
        self.fc = wc/2/np.pi

        sk = self.__poles()
        sk = sk[sk.real < 0]
        self.poles = sk

        tf = [0., 0.]
        tf[0], tf[1] = self.__tf()
        self.tf = tf
        self.tf_sos = self.__sos()

        tfz = [0., 0.]
        tfz[0], tfz[1] = self.__tfz()
        self.tfz = tfz
        self.tfz_sos = self.__sosz()
        

    def __corner(self, wp, Hwp, ws, Hws):
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
        log_wc = np.log10(self._A(Hwp))*np.log10(ws) - np.log10(self._A(Hws))*np.log10(wp)
        log_wc = log_wc/(np.log10(self._A(Hwp)/self._A(Hws)))

        return 10**log_wc


    def __order(self, wc, w, Hw):
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
        return np.log10(self._A(Hw))/(2*np.log10(w/wc))


    def __poles(self):
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
        N = self.order
        k = np.arange(2*N)
        return self.wc*np.exp((1j*np.pi/2/N)*(2*k + N - 1))


    def __tf(self):

        den = np.poly(self.poles).real
        num = np.array([den[-1]])

        return num, den


    def __tfz(self):

        sosz = futils.tf_to_sos_z(self.tf[0], self.tf[1], self.T)
        
        return futils.sos_to_tf(sosz[0], sosz[1])


    def __sos(self):

        return futils.tf_to_sos(self.tf[0], self.tf[1])


    def __sosz(self, T=None):

        T = T if T is not None else self.T
        
        return futils.tf_to_sos_z(self.tf[0], self.tf[1], T)
