import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pysp
#import scipy.signal

# --- Input ---
# Pass band
wp = 0.2*np.pi
Hwp = 0.89125
Hwp_dB = 20*np.log10(Hwp)

# Stop band
ws = 0.4*np.pi
Hws = 0.17783
Hwp_dB = 20*np.log10(Hwp)

# Freq
T = 1
fs = 1/T
fnyq = fs/2

# --- Filter ---
butter = pysp.filters.butter(wp, Hwp, ws, Hws, T=T, cutoff='wc', method='bilinear')


##wpl = 2/T*np.tan(wp*T/2)
##wsl = 2/T*np.tan(ws*T/2)
##
##A = lambda H_w : 1/H_w**2 - 1
##log_wc = np.log10(A(Hwp))*np.log10(wsl) - np.log10(A(Hws))*np.log10(wpl)
##log_wc = log_wc/(np.log10(A(Hwp)/A(Hws)))
##
##wcc = 10**log_wc

# --- Bilinear transf ---
##num = butter.tf[0]
##den = butter.tf[1]
###num = np.array([-4.55e15])
###den = np.array([1, 8215, 0, -5.544e11, -4.55e15])
##
##s, z = sp.symbols('s z')
##s = 2/T*(1 - 1/z)/(1 + 1/z)
##num_poly = sp.Poly(num[0], s)
##den_poly = sp.Poly(den, s)
##
##gs = num_poly/den_poly
##num_z, den_z = sp.fraction(gs.factor())
##num_z = sp.Poly(num_z.expand(), z)
##den_z = sp.Poly(den_z.expand(), z)
##
##num_z = num_z/den_z.coeffs()[0]
##den_z = den_z/den_z.coeffs()[0]
