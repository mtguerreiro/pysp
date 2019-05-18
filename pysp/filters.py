import numpy as np
import scipy.signal
import pysp.filter_utils as futils
import pysp.butter as spbutter
import sympy


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

    method : str
        Method used to design the digital filter.

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
    def __init__(self, wp, Hwp, ws, Hws, cutoff='wp', T=1, method='impulse'):

        self.method = method
        
        if method == 'impulse':
            self.__impulse(wp, Hwp, ws, Hws, cutoff, T)

        elif method == 'bilinear':
            self.__bilinear(wp, Hwp, ws, Hws, cutoff, T)

        elif method == 'fir':
            self.__fir(wp, Hwp, ws, Hws, T)


    def __impulse(self, wp, Hwp, ws, Hws, cutoff='wp', T=1):
            """Designs a Butterworth low-pass with the impulse method.

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
            self._A = lambda H_w : 1/H_w**2 - 1

            # Sampling time for discrete stuff
            self.T = T

            # Corner frequency and order
            _wc = spbutter.corner(wp, Hwp, ws, Hws)
            _N = spbutter.order(_wc, wp, Hwp)

            print(_N, _wc)
            
            # Actual order, cut-off
            N = np.ceil(_N).astype(int)
            if cutoff == 'wp':
                wc = wp/(self._A(Hwp)**(1/2/N))
            else:
                wc = ws/(self._A(Hws)**(1/2/N))
            self.order = N
            self.wc = wc
            self.fc = wc/2/np.pi

            # Poles
            sk = spbutter.poles(N, wc)
            sk = sk[sk.real < 0]
            self.poles = sk

            # Continuous-time transfer function
            tf = [0., 0.]
            tf[0], tf[1] = futils.tf_from_poles(sk)
            self.tf = tf
            self.tf_sos = futils.tf_to_parallel_sos(tf[0], tf[1])

            # Discrete-time transfer function
            tfz = [0., 0.]
            sosz = [0., 0.]
            sosz = futils.tf_to_parallel_sos_z(tf[0], tf[1], T)
            tfz = futils.parallel_sos_to_tf(sosz[0], sosz[1])
            self.tfz = tfz
            self.tfz_sos = sosz

            
    def __bilinear(self, wp, Hwp, ws, Hws, cutoff='wp', T=1):
            """Designs a Butterworth low-pass with the bilinear method.

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
            self._A = lambda H_w : 1/H_w**2 - 1

            # Sampling time for discrete stuff
            self.T = T

            # Corner frequency
            wpl = 2/T*np.tan(wp*T/2)
            wsl = 2/T*np.tan(ws*T/2)
            _wc = self.__corner(wpl, Hwp, wsl, Hws)
            # Order
            _N = self.__order(_wc, wpl, Hwp)
            
            # Actual order, cut-off
            N = np.ceil(_N).astype(int)
            if cutoff == 'wp':
                wc = wpl/(self._A(Hwp)**(1/2/N))
            else:
                wc = wsl/(self._A(Hws)**(1/2/N))
            self.order = N
            self.wc = wc
            self.fc = wc/2/np.pi

            sk = self.__poles()
            sk = sk[sk.real < 0]
            self.poles = sk

            tf = [0., 0.]
            tf[0], tf[1] = self.__tf()
            self.tf = tf
            #self.tf_sos = self.__sos()

            self.__bilinear_c_to_d()


    def __bilinear_c_to_d(self):

            T = self.T
            
            s, z = sympy.symbols('s z')
            s = 2/T*(1 - 1/z)/(1 + 1/z)

            num = self.tf[0]
            den = self.tf[1]
            num_poly = sympy.Poly(num[0], s)
            den_poly = sympy.Poly(den, s)

            gs = num_poly/den_poly
            num_z, den_z = sympy.fraction(gs.factor())
            num_z = sympy.Poly(num_z.expand(), z)
            den_z = sympy.Poly(den_z.expand(), z)

            num_z = num_z/den_z.coeffs()[0]
            den_z = den_z/den_z.coeffs()[0]
            num_z = sympy.Poly(num_z).all_coeffs()
            den_z = sympy.Poly(den_z).all_coeffs()
            
            self.tfz = [0., 0.]
            self.tfz[0] = np.array(num_z, np.float64)
            self.tfz[1] = np.array(den_z, np.float64)

            self.__bilinear_tfz_to_sos()


    def __bilinear_tfz_to_sos(self):

            numz = self.tfz[0]
            denz = self.tfz[1]

            zk = np.sort(np.roots(numz))
            pk = np.sort(np.roots(denz))

            den_list = []
            num_list = []
            k = 0
            while k < numz.shape[0] - 1:
                if np.abs(np.imag(zk[k])) > 1e-10:
                    numk = np.polymul([1, -zk[k]], [1, -zk[k+1]])
                    num_list.append(numk.real)
                    k += 1
                else:
                    num_list.append(np.poly(zk[k]).real)
                k += 1
            k = 0
            while k < denz.shape[0] - 1:
                if np.abs(np.imag(pk[k])) > 1e-10:
                    denk = np.polymul([1, -pk[k]], [1, -pk[k+1]])
                    den_list.append(denk.real)
                    k += 1
                else:
                    den_list.append(np.poly(pk[k]).real)
                k += 1

            self.tfz_sos = [0., 0.]
            self.tfz_sos[0] = np.array(num_list)
            self.tfz_sos[0][0] = numz[0]*self.tfz_sos[0][0]
            self.tfz_sos[1] = np.array(den_list)


    def __fir(self, wp, Hwp, ws, Hws, T):
        
        self.T = T
        dw = (ws - wp)*T
        wc = (ws + wp)*T/2
        delta = 1 - Hwp

        A = -20*np.log10(delta)

        if A < 21:
            beta = 0

        elif (A >= 21) and (A <= 50):
            beta = 0.5842*(A - 21)**0.4 + 0.07886*(A - 21)

        else:
            beta = 0.1102*(A - 8.7)

        M = (A - 8)/(2.285*dw)
        M = np.ceil(M).astype(int)
        
        self.A = A
        self.M = M
        
        alpha = M/2
        n = np.arange(M + 1)
        i0_arg = beta*(1 - ((n - alpha)/alpha)**2)**0.5
        w = scipy.special.i0(i0_arg)/scipy.special.i0(beta)
        self.w = w

        hpb = wc/np.pi*np.sinc(wc*(n - alpha)/np.pi)
        h = w*hpb
        self.hpb = hpb
        self.h = h

        tfz = [0.0, 0.0]
        tfz[0] = h
        tfz[1] = np.array([1.])
        self.tfz = tfz
        
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
        #sosz = futils.tf_to_sos_z(self.tf[0], self.tf[1], self.T)
        sosz = futils.tf_to_parallel_sos_z(self.tf[0], self.tf[1], self.T)  

        return futils.parallel_sos_to_tf(sosz[0], sosz[1])
    

    def __sos(self):
        """Gets SOS representation from the filter's transfer function.

        Returns
        -------
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

        return futils.tf_to_parallel_sos_z(self.tf[0], self.tf[1], T)


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
        ----------
        y : np.ndarray
            Filter's output.
            
        """
        if self.method == 'impulse':
            y = np.zeros(x.shape)
            for num, den in zip(self.tfz_sos[0], self.tfz_sos[1]):
                y += futils.sos_filter((num, den), x)
                
        elif self.method == 'bilinear':
            y = np.copy(x)
            for num, den in zip(self.tfz_sos[0], self.tfz_sos[1]):
                y = futils.sos_filter((num, den), y)

        elif self.method == 'fir':
            y = futils.filter_fir(self.tfz, x)
            
        return y
    

    def filter2(self, x):
        """Filters the signal :math:`x`.

        Parameters
        ----------
        x : np.ndarray
            1-D vector with filter's input.

        Returns
        ----------
        y : np.ndarray
            Filter's output.
            
        """
        if self.method == 'impulse':
            y = np.zeros(x.shape)
            for num, den in zip(self.tfz_sos[0], self.tfz_sos[1]):
                y += futils.sos_filter_rt((num, den), x)
            
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


        
