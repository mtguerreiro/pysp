import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pysp
#import scipy.signal

# --- Input ---
# Pass band
wp = 0.2*np.pi*10000
Hwp = 0.89125
Hwp_dB = 20*np.log10(Hwp)

# Stop band
ws = 0.4*np.pi*10000
Hws = 0.17783
Hwp_dB = 20*np.log10(Hwp)

# Freq
T = 1/10000
fs = 1/T
fnyq = fs/2

# Bode 
wstart = 0.05*wp/(2/np.pi)
wstop = 2*np.pi*fnyq
dw = 0.01

# Frequencies to simulate
freqs = [100, 1000, 4000]
N = 500

# Sample intervals to plot
sintervals = [[0, N], [0, 100], [0, 50]]

# --- Filter ---
w = np.arange(wstart, wstop, dw)
butter = pysp.filters.butter(wp, Hwp, ws, Hws, T=T)

Hw_mag, Hw_phase = butter.bode(w)
Hz_mag, Hz_phase = butter.bodez(w)

# --- Signals ---
f = 2*np.pi*np.array(freqs)*T
n = np.arange(N)
Nf = f.shape[0]
x = np.sin(f*n.reshape(-1,1))
y = np.zeros((N, Nf))
for k in range(Nf):
    y[:, k] = butter.filter(x[:, k])

# --- Plots ---
##matplotlib.rc('font', **{'family': 'serif'})
##matplotlib.rc('text', usetex=True)
##matplotlib.rcParams.update({'font.size': 12})

plt.ion()

plt.figure(figsize=(12, 9))
wnyq = 2*np.pi*fnyq
idx = np.where(w < wnyq)
plt.subplot(2,2,1)
plt.semilogx(w[idx]/2/np.pi, Hz_mag[idx], label='$H(z)$')
plt.semilogx(w/2/np.pi, Hw_mag, '--', label='$H(s)$')
plt.grid(True, which='both')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB)')
plt.title('Frequency response')
plt.legend()

for k in range(Nf):
    plt.subplot(2,2,k+2)
    plt.plot(x[:, k], label='Input')
    plt.plot(y[:, k], label='Output')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.xlim(sintervals[k])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('$f = ' + str(freqs[k]) + '$ Hz')
plt.tight_layout()
#plt.savefig('impulse-mag.pdf', bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.semilogx(w[idx]/2/np.pi, Hz_phase[idx], label='$H(z)$')
plt.semilogx(w/2/np.pi, Hw_phase, label='$H(s)$')
plt.grid(True, which='both')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Phase ($^\circ$)')
plt.tight_layout()
plt.legend()
#plt.savefig('impulse-phase.pdf', bbox_inches='tight')
