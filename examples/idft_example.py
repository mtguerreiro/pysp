import numpy as np
import matplotlib.pyplot as plt
import pysp
import time

# --- Input ---
# Number of frequency components
Nf = 1

# Sampling frequency
fs = 10e3

# Number of points
N = 2**10

# --- Signal ---
Nx = int(N/2)
fidx = np.random.randint(0, Nx/10, Nf)

X = np.zeros(N, dtype=complex)
X[fidx] = np.random.randint(-10, 10, Nf) + 1j*np.random.randint(-10, 10, Nf)
X[-fidx] = X[fidx]

fx = (fs/N)*np.arange(Nx)
Xm = np.abs(X[:Nx])

# --- IDFT ---
t = (1/fs)*np.arange(N)

_ti = time.time()
x = pysp.ft.dft_naive(X)
_tf = time.time()

# --- Output ---
# Plots
plt.ion()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fx/1e3, Xm)
plt.grid()
plt.title('Frequency domain')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(t/1e-3, x.real)
plt.grid()
plt.title('Time domain')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# Prints
print('Number of points:', N)
print('Execution time:', _tf - _ti)
