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

    show_partial : bool
        If True, prints mid-steps in design

    Attributes
    ----------
    order : int
        Filter's order.

    wc : float
        Filter's cut-off frequency, in rad/s.

    fc : float
        Filter's cut-off frequency, in Hz.

    poles : np.ndarray
        Filter's stable poles.

    tf : list
        Numerator and denominator of the filter's transfer function.

    tf_sos : list
        Numerator and denominator of the filter's transfer function cast in
        SOS mode.

    tfz : int
        Numerator and denominator of the filter's discrete transfer function.

    tfz_sos : list
        Numerator and denominator of the filter's discrete transfer function
        cast in SOS mode.
        
    T : int, float
        Sampling period for discrete filters.
        
    """
    def __init__(self, wp, Hwp, ws, Hws, cutoff='wp', T=1, show_partial=False,
                 method='impulse'):

        if method == 'impulse':
            self.__impulse(wp, Hwp, ws, Hws, cutoff, T, show_partial)
        

    def __impulse(self, wp, Hwp, ws, Hws, cutoff='wp', T=1, show_partial=False):
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
        """Computes the poles of a Butterworth filter with the specified
        cut-off frequency :math:`\omega_c` and order :math:`N`. 

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
        """Computes the numerator and denominator of the filter's transfer
        function based on its poles.

        Returns
        -------
        num : np.ndarray
            Array with numerator coefficients.

        den : np.ndarray
            Array with denominator coefficients.

        """
        den = np.poly(self.poles).real
        num = np.array([den[-1]])

        return num, den


    def __tfz(self):
        """Computes the numerator and denominator of the filter's discrete
        transfer function, based on the continuos-time transfer function.

        Returns
        -------
        num : np.ndarray
            Array with denominator coefficients.

        den : np.ndarray
            Array with numerator coefficients.
            
        """
        sosz = futils.tf_to_sos_z(self.tf[0], self.tf[1], self.T)        
        return futils.sos_to_tf(sosz[0], sosz[1])
    

    def __sos(self):
        """Gets SOS representation from the filter's transfer function.

        returns:
        num : np.ndarray
            Array with denominator coefficients for each section.

        den : np.ndarray
            Array with numerator coefficients for each section.
            
        """
        return futils.tf_to_sos(self.tf[0], self.tf[1])


    def __sosz(self, T=None):
        """Gets SOS representation from the filter's discrete transfer
        function.

        Returns
        -------
        num : np.ndarray
            Array with denominator coefficients for each section.

        den : np.ndarray
            Array with numerator coefficients for each section.
            
        """
        T = T if T is not None else self.T

        return futils.tf_to_sos_z(self.tf[0], self.tf[1], T)


    def bode(self, w):
        """Gets the magnitude and phase for the filter's transfer function,
        over the frequency range specified.

        Parameters
        ----------
        w : np.ndarray
            Vector with frequencies to evaluate the filter's magnitude and
            frequency response.

        Returns
        -------
        mag : np.ndarray
            Array with magnitudes for the response of the filter's transfer
            function.
        phase : np.ndarray
            Array with phases for the response of the filter's transfer
            function.

        """
        # --- Method 1 ---
        _, H_mag, H_phase = scipy.signal.bode((self.tf[0], self.tf[1]), w)

        # --- Method 2 ---
        #_, H = scipy.signal.freqs(self.tf[0], self.tf[1], w)
        #H_mag = np.abs(H)
        #H_phase = np.angle(H)

        # --- Method 3 ---
        #H_n = np.polyval(self.tf[0], 1j*w)
        #H_d = np.polyval(self.tf[1], 1j*w)
        #H = H_n/H_d

        #H_mag = np.abs(H)
        #H_phase = np.angle(H)
        
        return H_mag, H_phase


    def bodez(self, w):
        """Gets the magnitude and phase for the discrete filter's transfer
        function, over the frequency range specified.
        """
        
        T = self.T

        # --- Method 1 ---
        #ejw = np.exp(-1j*w*T)

        #Hz_n = np.polyval(self.tfz[0], ejw)
        #Hz_d = np.polyval(self.tfz[1], ejw)
        #Hz = Hz_n/Hz_d
        
        #Hz_mag = np.abs(Hz)
        #Hz_phase = np.angle(Hz)

        # --- Method 2 ---
        _, Hz_mag, Hz_phase = scipy.signal.dbode((self.tfz[0], self.tfz[1], T), w*T)

        # --- Method 3 ---
        #_, Hz = scipy.signal.freqz(self.tfz[0], self.tfz[1], w, fs=2*np.pi*1/T)
        #Hz_mag = np.abs(Hz)
        #Hz_phase = np.angle(Hz)
        
        return Hz_mag, Hz_phase
    

    def evaluate(self, w):
        """Evaluates the filter's frequency response at the specified
        frequency or frequencies.

        Parameters
        ----------
        w : int, float, np.ndarray
            Frequency to evaluate the filter's response.
            
        Returns
        -------
        int, float, np.ndarray
            Filter's complex response at the specified frequency of
            frequencies.
            
        """
        return np.polyval(self.tf[0], 1j*w)/np.polyval(self.tf[1], 1j*w)


    def evaluatez(self, w):
        """Evaluates the discrete filter's frequency response at the specified
        frequency or frequencies. The frequency range should not be normalized
        to the :math:`[\pi, \pi)` interval.
 
        Parameters
        ----------
        w : int, float, np.ndarray
            Frequency to evaluate the filter's response
            
        Returns
        -------
        int, float, np.ndarray
            Filter's complex response at the specified frequency of
            frequencies.
            
        """
        T = self.T
        ejw = np.exp(-1j*w*T)
        
        return np.polyval(self.tfz[0], ejw)/np.polyval(self.tfz[1], ejw)


    def filter(self, x):
        """Filters the signal :math:`x`.

        Parameters
        ----------
        x : np.ndarray
            1-D vector with filter's input.

        Returns
        y : np.ndarray
            Filter's output.
            
        """
        y = np.zeros(x.shape)
        for num, den in zip(self.tfz_sos[0], self.tfz_sos[1]):
            y += futils.sos_filter((num, den), x)

        return y
    

##    def filterz(self, x):
##        """Filters the signal :math:`x`.
##
##        Parameters
##        ----------
##        x : np.ndarray
##            Filter's input, as a 1-D array.
##
##        Returns
##        y : np.ndarray
##            Filter's output, as a 1-D array
##            
##        """
##        N = x.shape[0]
##        M = self.order
##
##        num = self.tfz[0][::-1]
##        den = self.tfz[1][1:][::-1]
##
##        Nn = num.shape[0]
##        Nd = den.shape[0]
##        
##        y = np.zeros(x.shape)
##
##        for n in range(0, M):
##            yi = np.where((n - np.arange(Nn)) >= 0)[0]
##            xi = np.where((n - np.arange(Nd)) >= 0)[0]
##            y[n] = -y[yi] @ den[yi] + x[xi] @ num[xi]
##            
##        for n in range(M, N):
##            y[n] = -y[(n - Nn):n] @ den + x[(n - Nd):n] @ num
##
##        return y
