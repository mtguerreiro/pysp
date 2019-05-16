import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pysp
#import scipy.signal

# --- Input ---
# Pass band
#wp = 2*np.pi*1000
#Hwp = 0.89125
wp = 0.2*np.pi*10000
Hwp = 1 - 0.01
Hwp_dB = 20*np.log10(Hwp)

# Stop band
#ws = 2*np.pi*1500
#Hws = 0.17783
ws = 0.4*np.pi*10000
Hws = 0.01
Hwp_dB = 20*np.log10(Hwp)

# Freq
T = 1/10000
fs = 1/T
fnyq = fs/2

# Bode 
wstart = 0.025*wp/(2/np.pi)
wstop = 2*np.pi*fnyq
dw = 0.01

# --- Filter ---
butter = pysp.filters.butter(wp, Hwp, ws, Hws, T=T, cutoff='wp', method='fir')

w = np.arange(wstart, wstop, dw)
Hz_mag, Hz_phase = butter.bodez(w)
#Hw_mag, Hw_phase = butter.bode(w)

x = np.sin(0.1*np.pi*np.arange(1000))

##fir = butter.tfz
##
##num = fir[0]
##
##N = x.shape[0]
##M = butter.M
##
### Append initial zeros to compensate for initial conditions
##xt = np.hstack((np.zeros(M), x))
##
##y = np.zeros(xt.shape)
##
##for n in range(M, N + M):
##    y[n] = xt[(n - M):(n + 1)] @ num
##    #y[n] = x[n]*num[0] + x[(n-M):n] @ num[1:][::-1]
##
##y = y[M:]

y = butter.filter(x)

#N = x.shape[0]
##numz = butter.tfz[0]
##denz = butter.tfz[1]
#numz = butter.tfz_sos[0][1]
#denz = butter.tfz_sos[1][1]
#y = np.zeros(x.shape)

##for n in range(N):
##    y[n] = -denz[1]*y[n-1] +\
##           numz[0]*x[n] + numz[1]*x[n-1]
##for n in range(N):
##    y[n] = -denz[1]*y[n-1] + -denz[2]*y[n-2] +\
##           numz[0]*x[n] + numz[1]*x[n-1] + numz[2]*x[n-2]
##for n in range(N):
##    y[n] = -denz[1]*y[n-1] + -denz[2]*y[n-2] + -denz[3]*y[n-3] +\
##           numz[0]*x[n] + numz[1]*x[n-1] + numz[2]*x[n-2] + numz[3]*x[n-3]    

# --- Output ---
plt.ion()

plt.figure()
plt.plot(x)
plt.plot(y)

#plt.figure()
#plt.semilogx(w/2/np.pi, 10**(Hz_mag/20))
#plt.semilogx(w/2/np.pi, 10**(Hw_mag/20))
plt.figure()
plt.semilogx(w/2/np.pi, Hz_phase)
