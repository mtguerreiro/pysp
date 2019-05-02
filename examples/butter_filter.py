import numpy as np
import matplotlib.pyplot as plt
import pysp

# --- Input ---
fp = 1000
Hfp = 0.89125

fs = 1500
Hfs = 0.17783

T = 1/10000

fstart = 100
fstop = 15000
df = 0.1

# --- Definitions ---
wp = 2*np.pi*fp
ws = 2*np.pi*fs
w = 2*np.pi*np.arange(fstart, fstop, df)

# --- Filter ---
butter = pysp.filters.butter(wp, Hfp, ws, Hfs, T=T, show_partial=True)

# --- Frequency responses ---
Hw_mag, Hw_phase = butter.bode(w)
Hz_mag, Hz_phase = butter.bodez(w)

# --- Plots ---
plt.ion()

plt.figure()
plt.semilogx(w/2/np.pi, Hw_mag)
plt.semilogx(w/2/np.pi, Hz_mag, '--')
plt.grid()

plt.figure()
plt.semilogx(w/2/np.pi, Hw_phase)
plt.semilogx(w/2/np.pi, Hz_phase, '--')
plt.grid()

