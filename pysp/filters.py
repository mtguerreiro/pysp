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

    cutoff : str
        Defines whether the filter should meet exactly the pass-band
        (:math:`\omega_p`) or stop-band (:math:`\omega_s`) frequency response.

    T : int, float
        Sampling time to design the discrete filter.

    method : str
        Method used to design the digital filter. The methods available are
        'impulse', 'bilinear' or 'fir'. 

    Attributes
    ----------
    order : int
        Filter's order.

    wc : float
        Filter's cut-off frequency, in rad/s. Only valid for filters with the
        impulse and bilinear methods.

    fc : float
        Filter's cut-off frequency, in Hz. Only valid for filters with the
        impulse and bilinear methods.

    poles : np.ndarray
        Filter's stable poles. Only valid for filters with the impulse and
        bilinear methods.

    tf : list
        Numerator and denominator of the filter's continuous-time transfer
        function. Only valid for filters with the impulse and bilinear
        methods.

    tf_sos : list
        Numerator and denominator of the filter's continuous-time transfer
        function cast in SOS mode. Only valid for filters with the impulse
        and bilinear methods. For the impulse method, this is the transfer
        function factored as a sum of second-order terms. For the bilinear
        method, this is the transfer function factored as a product of
        second-order terms.

    tfz : int
        Numerator and denominator of the filter's discrete transfer function.

    tfz_sos : list
        Numerator and denominator of the filter's discrete transfer function
        cast in SOS mode. For the impulse method, this is the transfer
        function factored as a sum of second-order terms. For the bilinear
        method, this is the transfer function factored as a product of
        second-order terms.
        
    T : int, float
        Sampling period for discrete filters.

    w : np.ndarray
        Kaiser window. Only valid for the FIR method.

    hpb : np.ndarray
        Ideal low-pass filter impulse response. Only valid for the FIR method.

    h : np.ndarray
        Impulse response for the FIR filter. Only valid for the FIR method.
        
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

        cutoff : str
            Defines whether the filter should meet exactly the pass-band
            (:math:`\omega_p`) or stop-band (:math:`\omega_s`) frequency
            response.

        T : int, float
            Sampling time to design the discrete filter.
            
        """
        A = lambda H_w : 1/H_w**2 - 1

        # Sampling time for discrete stuff
        self.T = T

        # Corner frequency and order
        _wc = spbutter.corner(wp, Hwp, ws, Hws)
        _N = spbutter.order(_wc, ws, Hws)

        print(_N, _wc)
        
        # Actual order, cut-off
        N = np.ceil(_N).astype(int)
        if cutoff == 'wp':
            wc = wp/(A(Hwp)**(1/2/N))
        else:
            wc = ws/(A(Hws)**(1/2/N))
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

        cutoff : str
            Defines whether the filter should meet exactly the pass-band
            (:math:`\omega_p`) or stop-band (:math:`\omega_s`) frequency response.

        T : int, float
            Sampling time to design the discrete filter.
            
        """
        A = lambda H_w : 1/H_w**2 - 1

        # Sampling time for discrete stuff
        self.T = T

        # Pre-warping
        wpl = 2/T*np.tan(wp*T/2)
        wsl = 2/T*np.tan(ws*T/2)

        # Corner frequency and order
        _wc = spbutter.corner(wpl, Hwp, wsl, Hws)
        _N = spbutter.order(_wc, wpl, Hwp)
        
        # Actual order, cut-off
        N = np.ceil(_N).astype(int)
        if cutoff == 'wp':
            wc = wpl/(A(Hwp)**(1/2/N))
        else:
            wc = wsl/(A(Hws)**(1/2/N))
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

        # Discrete-time transfer function
        self.tfz = [0., 0.]
        tfz = futils.bilinear_transform(self.tf[0], self.tf[1], T)
        self.tfz[0], self.tfz[1] = tfz[0], tfz[1]
            
        self.tfz_sos = [0., 0.]
        sosz = futils.tf_to_cascaded_sos(tfz[0], tfz[1])
        self.tfz_sos[0], self.tfz_sos[1] = sosz[0], sosz[1]
        

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
        
        self.order = M
        
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
