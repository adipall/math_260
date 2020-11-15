# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:16:10 2020
Project - Milestone 2
@author: Adi
"""
import numpy as np
from numpy import fft
from scipy.signal import wiener as wien
from matplotlib import pyplot as plt

if __name__ == "__main__":

    np.random.seed(5)
    N = 2000
    dt = 0.05
    
    t = dt * np.arange(N)
    signal = np.exp(-0.2 * (t - 30.) ** 2)
    noise = np.random.normal(0, 0.1, size=signal.shape) # white noise
    
    comb = signal + noise
    
    # signal in the time domain
    plt.figure(figsize = (6.5,2.5))
    plt.plot(t,signal)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Signal in Time Domain')
    plt.show()    
    
    # signal + noise in the time domain
    plt.figure(figsize = (6.5,2.5))
    plt.plot(t,comb)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Combination in Time Domain')
    plt.show()    
    
    # signal in frequency domain
    transform = fft.fft(signal)
    freq = fft.fftfreq(signal.shape[0], dt)
    freq = fft.fftshift(freq)
    transform = fft.fftshift(transform)
    S = np.abs(transform)
    S_sq = (np.abs(transform))**2
    plt.loglog(freq, S)
    plt.ylabel('|F|')
    plt.xlabel('Freq(Hz)')
    plt.title('Signal in Frequency Domain')
    plt.show()    
    
    # signal + noise in frequency domain with noise estimation
    transform = fft.fft(comb)
    freq = fft.fftfreq(comb.shape[0], dt)
    freq = fft.fftshift(freq)
    transform = fft.fftshift(transform)
    C_re_im = transform
    C = np.abs(transform)
    C_sq = (np.abs(transform))**2
    plt.figure(figsize = (6.5,4.5))
    plt.loglog(freq, C)
    # create horizontal line of very simple noise estimate
    noise_estim = np.full(2000,5)
    plt.loglog(freq,noise_estim)
    plt.ylabel('|F|')
    plt.xlabel('Freq(Hz)')
    plt.legend(['combined', 'noise estimate'])
    plt.title('Combination in Frequency Domain')
    plt.show()    

    phi = 1/(1+(noise_estim)/S) # 'Optimal Wiener Filter'
    filt_S = phi*C_re_im # applying filter 

    plt.figure(figsize = (6.5,4.5))
    plt.loglog(freq,np.abs(filt_S))
    plt.ylabel('|F|')
    plt.xlabel('Freq(Hz)')
    plt.title('Filtered Signal in Frequency Domain')
    plt.show()
    
    filt_S = fft.fftshift(filt_S)
    filt_s = fft.ifft(filt_S)
    plt.figure(figsize=(6.5,4.5))
    plt.plot(t,filt_s)
    plt.ylabel('Amp')
    plt.xlabel('time (s)')
    plt.title('Filtered Signal using Crude Noise Estimate (Wiener)')
    
    # just for comparison to scipy result
    filtered = wien(comb,mysize=30)
    plt.figure(figsize = (6.5,2.5))
    plt.plot(t,filtered)    
    plt.ylabel('Amp')
    plt.xlabel('time (s)')
    plt.title('Filtered Signal using Scipy Wiener Filter')
    
    # why does mine look nicer? will be coming to OH monday
    

