import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pysp
import pysp.filter_utils as futils
import scipy.signal

# --- Input ---
fc = 100

T = 1/10000

num = [1]
den = [1, 1.4142, 1]

# Bode 
wstart = 10/(2/np.pi)
wstop = 5000/(2/np.pi)
dw = 1

# --- Discretization ---
wc = 2*np.pi*fc

num = np.array([num[0]*wc**2])
den = np.array([den[0], den[1]*wc, den[2]*wc**2])

sosz = [0., 0.]
sosz = futils.tf_to_parallel_sos_z(num, den, T)
numz, denz = futils.parallel_sos_to_tf(sosz[0], sosz[1])

numzbl, denzbl = pysp.filter_utils.bilinear_transform(num, den, T)


# --- Phase response ---
w = np.arange(wstart, wstop, dw)

_, Hz_mag, Hz_phase = scipy.signal.dbode((numz, denz, T), w*T)
_, Hz_mag_bl, Hz_phase_bl = scipy.signal.dbode((numzbl, denzbl, T), w*T)
_, Hs_mag, Hs_phase = scipy.signal.bode((num, den), w)

# --- Output ---
plt.ion()

plt.semilogx(w/2/np.pi, Hz_mag)
plt.semilogx(w/2/np.pi, Hz_mag_bl, '--')
plt.semilogx(w/2/np.pi, Hs_mag, '--')
plt.grid()
