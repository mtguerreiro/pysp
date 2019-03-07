import numpy as np
import matplotlib.pyplot as plt
import pysp
import time

# --- Input ---
f = 100
fs = 100e3

#N = 3*int(fs/f)
N = 2**12

# --- Signal ---
t = (1/fs)*np.arange(N)

x = np.sin(300*2*np.pi*f*t)

# --- DFT ---
Nx = int(N/2)
fx = (fs/N)*np.arange(Nx)

_ti = time.time()
X = pysp.ft.dft_naive(x)
_tf = time.time()
print(_tf - _ti)

Xm = np.abs(X[:Nx])

# --- Output ---
plt.ion()
plt.figure()
plt.plot(t, x)

plt.figure()
plt.plot(fx, Xm)
