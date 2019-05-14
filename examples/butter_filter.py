import numpy as np
import scipy.signal
import sympy as sp
import pysp
import matplotlib.pyplot as plt

# --- Input ---
wp = 2*np.pi*1000
Hwp = 0.89125
Hwp_dB = 20*np.log10(Hwp)

ws = 2*np.pi*1500
Hws = 0.17783
Hwp_dB = 20*np.log10(Hwp)

T = 1/10000

fstart = 0.01*wp/(2/np.pi)
fstop = ws/(2/np.pi)
df = 0.01

# --- Filter ---
w = 2*np.pi*np.arange(fstart, fstop, df)
butter = pysp.filters.butter(wp, Hwp, ws, Hws, T=T, show_partial=True)

Hw_mag, Hw_phase = butter.bode(w)
Hz_mag, Hz_phase = butter.bodez(w)

# --- Signal ---
x = np.sin(0.3*np.arange(1000))

y = butter.filter(x)
# --- Implementation ---
##N = x.shape[0]
##M = butter.order
##
##num = butter.tfz[0][::-1]
##den = butter.tfz[1][1:][::-1]
##
##Nn = num.shape[0]
##Nd = den.shape[0]
##
##y = np.zeros(x.shape)
##for n in range(0, M):
##    yi = np.where((n - 1 - np.arange(Nn)) >= 0)[0]
##    xi = np.where((n - np.arange(Nd)) >= 0)[0]
##    y[n] = -y[yi] @ den[yi] + x[xi] @ num[xi]
##
##for n in range(M, N):
##    y[n] = -y[(n - Nd):n] @ den + x[(n - Nn):n] @ num

# --- Scipy filter ---
##numz = np.hstack((butter.tfz_sos[0], np.zeros((3, 1))))
##denz = butter.tfz_sos[1]
##sosz = np.hstack((numz, denz))
##y_sp = scipy.signal.sosfilt(sosz, x)
#num_sp, den_sp = scipy.signal.butter(butter.order, butter.wc*T)
#y_sp = scipy.signal.lfilter(num_sp, den_sp, x)

# --- Plots ---
#plt.ion()
plt.plot(x, label='Input')
plt.plot(y, label='Input')

plt.show()
##plt.plot(y, label='Implementation')
##plt.plot(y_sp, '--', label='Scipy')

#plt.figure()
#plt.semilogx(w/2/np.pi, Hw_mag)

#plt.figure()
#plt.semilogx(w/2/np.pi, Hw_phase)
