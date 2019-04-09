import numpy as np
import csv
import re

def samples_bit_reversal(x):

    N = int(x.shape[0])
    n = int(np.log10(N)/np.log10(2))
    
    x = ( x & 0x55555555 ) << 1 | ( x & 0xAAAAAAAA ) >> 1
    x = ( x & 0x33333333 ) << 2 | ( x & 0xCCCCCCCC ) >> 2
    x = ( x & 0x0F0F0F0F ) << 4 | ( x & 0xF0F0F0F0 ) >> 4
    x = ( x & 0x00FF00FF ) << 8 | ( x & 0xFF00FF00 ) >> 8
    x = ( x & 0x0000FFFF ) << 16 | ( x & 0xFFFF0000 ) >> 16

    return x >> (32 - n)


def samples_bit_reversal_old(N):

    if type(N) is not int:
        raise TypeError('``N`` must be an integer')

    n = np.log10(N)/np.log10(2)
    fmt = '0' + str(int(n) + 2) + 'b'
    
    return [int(format(xi, fmt)[:1:-1], 2) for xi in np.arange(N)]


def butterfly_idx(nb, mb):

    alp = [np.arange(nb) + 2*nb*k for k in range(int(mb))]
    bet = [np.arange(nb) + 2*nb*k + nb for k in range(int(mb))]

    return alp, bet


def butterfly_idx2(nb, mb):
    
    alp = []
    bet = []
    for k in range(mb):
        x = np.arange(nb) + 2*nb*k 
        alp.append(x.tolist())
        bet.append((x + nb).tolist())

    return alp, bet
    

def butterfly(x, y, beta):

    return x+beta*y, x-beta*y


def ltspice_read_txt(file):

    with open(file) as csvfile:
        csvdata = list(csv.reader(csvfile, delimiter='\t'))
        traces = csvdata[0]
        traces_data = np.array([[float(dk) for dk in d] for d in csvdata[1:]])

    data = {}
    for k, trace in enumerate(traces):
        data[trace] = traces_data[:, k]

    return data


def ltspice_read_bode(file):

    with open(file, 'r') as csvfile:
        csvdata = list(csv.reader(csvfile, delimiter='\t'))

    traces = csvdata[0]
    freq = []
    mag_phase = []
    for k, d in enumerate(csvdata[1:]):
        freq.append(float(d[0]))
        mp = re.findall('(-*\d.\d*e[+-]\d*)', d[1])
        mag_phase.append([float(mi) for mi in mp]) 

    data = {}
    data['freq'] = np.array(freq)
    data['mag'] = np.array(mag_phase)[:, 0]
    data['phase'] = np.array(mag_phase)[:, 1]

    return data
