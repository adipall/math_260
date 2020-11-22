# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:16:10 2020
@author: Adi
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from numpy import gradient
from numpy import fft
# from scipy.signal import wiener as wien

def tanh_fit(x, A, B, x0, sigma):
    """ hyperbolic tangent function for power spectrum fit """
    return A+B*np.tanh((x-x0)/sigma)

def find_horiz(func, x , y, vis_fit = True):
    """ determine flat noise estimate (no time dependence of noise) """
    x_space = np.linspace(start=min(x), stop=max(x), num=100)
    pars, cov = curve_fit(f=tanh_fit, xdata=x, ydata=y)
    y_fit = tanh_fit(x_space,*pars)
    if vis_fit:      
        plt.loglog(x_space,y_fit)
    dfit = gradient(y_fit)
    zero_loc = [ i for i,x in enumerate(dfit) if x.any() == 0 ]
    val = np.mean(y_fit[zero_loc])
    print(f"|N|**2 approx = {val}")
    y_flat = np.full(len(x),val)
    return y_flat

def gen_noisy_imp(N,dt,shift):
    """ generate test signal and combination for simple impulse """
    np.random.seed(5)
    t = dt * np.arange(N)
    signal = np.exp(-0.2 * (t - shift) ** 2)
    noise = np.random.normal(0, 0.1, size=signal.shape) # white noise   
    comb = signal + noise
    return t, signal, comb

def get_noise(freq, P, wind_num,vis_fit = True):
    """ plots each window's power spectrum separately and returns noise """
    yflat = find_horiz(tanh_fit,freq,P,vis_fit)
    if vis_fit:
        plt.loglog(freq, P)
        plt.loglog(freq,yflat)
        plt.ylabel('|F|**2')
        plt.xlabel('Freq(Hz)')
        plt.title(f'window {wind_num}')
        plt.show()
    noise_est = yflat
    return noise_est

def window_wien(data,num_w,dt):
    """ Uses hanning window to get power spectrum for num_w windows and 
        determines noise estimate at each. Filters each window
        to return an average filtered signal in frequency domain with
        num_w/2 times smaller resolution (terrible method clearly) """
    N = data.shape[0]
    nfft = N//num_w
    k = np.arange(0,nfft//2)
    f = k/(dt*nfft)
    w = np.hanning(nfft)
    window = w*np.sqrt(nfft/sum(w**2))
    sigMat = data.reshape((nfft,num_w))
    X_sq_sum = np.zeros(nfft//2)
    filt_X_sq_sum = np.zeros(nfft//2)
    filt_X_sum = np.zeros(nfft//2)
    sum_est = 0
    for i in range(num_w):
        cur = sigMat[:,i]
        xw = cur*window
        X = fft.fft(xw)
        X = X[0:nfft//2] # grab only FFT corresponding to 0 to fs/2
        X_sq = abs(X)**2
        noise_est = get_noise(f,X_sq,i,vis_fit = True) # plot each and get noise
        phi = 1/(1+(noise_est)/(X-noise_est)) # 'Optimal Wiener Filter'
        filt_X = phi*X # applying filter
        filt_X_sq = abs(filt_X)**2
        # track sums
        X_sq_sum = X_sq_sum + X_sq
        filt_X_sum = filt_X_sum + filt_X
        filt_X_sq_sum = filt_X_sq_sum + filt_X_sq
        sum_est = sum_est + noise_est[0] # for horizontal only
    X_sq_avg = X_sq_sum/num_w
    filt_X_sq_avg = filt_X_sq_sum/num_w
    filt_X_avg = filt_X_sum/num_w
    avg_est = sum_est/num_w
    return f, X_sq_avg, filt_X_sq_avg, filt_X_avg, avg_est
    
if __name__ == "__main__":
    
    # generate noisy impulse signal
    dt = 0.025 # 1/fs
    num_pts = 4000
    shift = 30
    t, signal, comb = gen_noisy_imp(num_pts, dt, shift)
    num_w = 8
    
    # signal in the time domain
    plt.figure(figsize = (6.5,2.5))
    plt.plot(t,signal)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Signal in Time Domain')
    plt.show()    
    
    # signal + noise in the time domain
    fig1 = plt.figure(figsize = (6.5,2.5))
    plt.plot(t,comb)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Combination in Time Domain')
    plt.show()
    fig1.savefig('combined.png')
    
    # can stucture better (reduce returns) after determine method is correct
    freq, C_sq, filtC_sq, filtC, avg_noise = window_wien(comb,num_w,dt)
    
    plt.loglog(freq, C_sq)
    yflat = find_horiz(tanh_fit,freq,C_sq)
    plt.loglog(freq, yflat)
    plt.loglog(freq, filtC_sq)
    plt.legend(['combined','fit','horiz','filtered'])
    plt.ylabel('|F|**2')
    plt.xlabel('Freq(Hz)')
    plt.title('Signal in Frequency Domain')
    plt.show()   
    
    # method 1 - filtered during windowing (leading to resolution decrease)
    plt.figure(figsize = (6.5,2.5))
    t_new = np.linspace(start=min(t), stop=max(t), num=250) # shouldn't have to do this
    filt_c = fft.ifft(filtC)
    # result is recognizable with the exception of the first index
    # also, the resolution decreases from 4000 -> 250 which seems awful
    # I am definitely doing something wrong
    filt_c[0] = 0   # why is this happening?
    plt.plot(t_new,filt_c)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Combination in Time Domain')
    plt.show()    
    
    # method 2 - using an average of noise estimates taken from each window
    # get a much better result this way, but would only work (in its current state
    # with the resolution issue) for flat and simple time dependent noise, 
    # because these can be easily extended to full frequency range
    
    transform = fft.fft(comb)
    freq = fft.fftfreq(comb.shape[0], dt)
    freq = fft.fftshift(freq)
    C_re_im = fft.fftshift(transform)
    C_sq = np.abs(C_re_im)**2
    
    transform = fft.fft(signal)
    S_sq = np.abs(fft.fftshift(transform))**2
    
    noise_estim = np.full(S_sq.shape[0],avg_noise)
    # NOW THIS PART IS ONLY WORKING IF I WERE TO KNOW SIGNAL!!! I don't see why
    phi = 1/(1+(noise_estim)/S_sq)
    # phi = 1/(1+(noise_estim)/(C_sq - noise_estim)) <---- should be this
    filt_C = phi*C_re_im # applying filter 

    plt.figure(figsize = (6.5,4.5))
    plt.loglog(freq,np.abs(filt_C))
    plt.ylabel('|F|')
    plt.xlabel('Freq(Hz)')
    plt.title('Filtered Signal in Frequency Domain')
    plt.show()
    
    filt_C = fft.fftshift(filt_C)
    filt_c = fft.ifft(filt_C)
    fig3 = plt.figure(figsize=(6.5,4.5))
    plt.plot(t,filt_c)
    plt.ylabel('Amp')
    plt.xlabel('Time (s)')
    plt.title('Filtered Signal using Average Flat Noise Estimate of 8 Windows')
    fig3.savefig('filtered_signal_v2.png')
    
    # # just for comparison to scipy result
    # filtered = wien(comb, mysize = 30)
    # plt.figure(figsize = (6.5,2.5))
    # plt.plot(t,filtered)    
    # plt.ylabel('Amp')
    # plt.xlabel('Time (s)')
    # plt.title('Filtered Signal using Scipy Wiener Filter')
