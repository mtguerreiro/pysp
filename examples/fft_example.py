import numpy as np
import matplotlib.pyplot as plt
import pysp
import time

# --- Input ---
# Signal's frequency
f = 100

# Sampling frequency
fs = 10e3

# Number of points
#N = 3*int(fs/f)
N = 2**12

# --- Signal ---
t = (1/fs)*np.arange(N)

x = np.sin(2*np.pi*f*t)

# --- DFT ---
Nx = int(N/2)
fx = (fs/N)*np.arange(Nx)

_ti = time.time()
#X = np.fft.fft(x)
#X = pysp.ft.fft(x)
X = pysp.ft.fft_recursive(x)
_tf = time.time()

Xm = np.abs(X[:Nx])

# --- Output ---
# Plots
plt.ion()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t/1e-3, x)
plt.grid()
plt.title('Time domain')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(fx/1e3, Xm)
plt.grid()
plt.title('Frequency domain')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')

# Prints
print('Number of points:', N)
print('Execution time:', _tf - _ti)
