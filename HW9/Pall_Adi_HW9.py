""" 
Adi Pall - HW 9 - Math 260
10/29/20
"""

# DIAL1: 555-3429
# DIAL2: 800-6284
# NOISY DIAL: 555-3429

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy.signal import find_peaks


def tone_data():
    """ Builds the data for the phone number sounds...
        Returns:
            tones - list of the freqs. present in the phone number sounds
            nums - a dictionary mapping the num. k to its two freqs.
            pairs - a dictionary mapping the two freqs. to the nums

        Each number is represented by a pair of frequencies: a 'low' and 'high'
        For example, 4 is represented by 697 (low), 1336 (high),
        so nums[4] = (697, 1336)
        and pairs[(697, 1336)] = 4
    """
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]  # (Hz)

    nums = {}
    for k in range(0, 3):
        nums[k+1] = (lows[k], highs[0])
        nums[k+4] = (lows[k], highs[1])
        nums[k+7] = (lows[k], highs[2])
    nums[0] = (lows[1], highs[3])

    pairs = {}
    for k, v in nums.items():
        pairs[(v[0], v[1])] = k

    tones = lows + highs  # combine to get total list of freqs.
    return tones, nums, pairs


def load_wav(fname):
    """ Loads a .wav file, returning the sound data.
        If stereo, converts to mono by averaging the two channels

        Returns:
            rate - the sample rate (in samples/sec)
            data - an np.array (1d) of the samples.
            length - the duration of the sound (sec)
    """
    rate, data = wavfile.read(fname)
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:, 0] + data[:, 1]  # stereo -> mono
    length = data.shape[0] / rate
    print(f"Loaded sound file {fname}.")
    return rate, data, length

def real_imag(freq, F, name):
    """ Plots the real and imaginary components of transform
        
        Inputs: frequency array (Hz), transform array, name of file
    """
    plt.figure(figsize = (6.5,2.5))
    plt.suptitle(name)
    plt.subplot(1,2,1)
    plt.loglog(freq, np.real(F), '.k')
    plt.ylabel('Re(F)')
    plt.subplot(1,2,2)
    plt.loglog(freq, np.imag(F), '.k')
    plt.ylabel('Imag(F)')
    plt.xlabel('Freq(Hz)')
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
def mag_plot(freq, abs_trans, fname):
    """ Plots the magnitude of transform
        
        Inputs: frequency array (Hz), Mag(transform array), name of file
    """
    plt.loglog(freq, abs_trans, '.k')
    plt.ylabel('Mag(F)')
    plt.xlabel('Freq(Hz)')
    plt.title(fname)
    plt.show()    
    
def dft(fname):
    """ Performs transform and plots real and imaginary parts
        Input: wav file name
        Returns: freq array, transform array
    """
    rate, data, length = load_wav(fname)
    # take fft of "data"...
    transform = fft.fft(data)
    freq = fft.fftfreq(data.shape[0], 1/rate)
    freq = fft.fftshift(freq)
    transform = fft.fftshift(transform)
    # check k/L rule:
    print(data.shape[0]/length == rate)
    # Ans: True
    real_imag(freq, transform, "DFT (real & imaginary parts)")
    return freq, transform
    
def dft_mag(fname):
    """ Performs transform and plots real and imaginary parts
        Input: wav file name
        Returns: freq array, Mag(transform array)
    """
    rate, data, length = load_wav(fname)
    tones, nums, pairs = tone_data()
    # take fft of "data"...
    transform = fft.fft(data)
    freq = fft.fftfreq(data.shape[0], 1/rate)
    freq = fft.fftshift(freq)
    transform = fft.fftshift(transform)
    abs_trans = np.abs(transform)
    mag_plot(freq,abs_trans,fname)
    # real_imag(data, np.abs(transform), "DFT (real & imaginary parts)")
    return freq, abs_trans

def identify_digit(fname, prom = 10E6):
    """ Uses find_peaks() to determine frequencies at local maxima
        Matches these frequencies to a digit using pairs dictionary
        Input: wav file name, prominence of peaks we seek (default = 10E6)
    """
    tones, nums, pairs = tone_data()
    freq, abs_F = dft_mag(fname)
    # frequency list contains all of the frequencies, so now just need the freqs @ peak
    pks_loc = find_peaks(abs_F, prominence=prom)
    pks_loc = pks_loc[0][2:4] # select out last two, since first two are negative versions
    # of same frequency
    freq1 = int(round(abs(freq[pks_loc[0]])))
    freq2 = int(round(abs(freq[pks_loc[1]])))
    
    if len(pks_loc) == 2: # if two frequencies were not found
        digit = pairs[(freq1, freq2)]
        print(f'Digit found: {digit}')
        return digit # grabs digit corresponding to freq pair
    else:
        print('no frequency pair to match to digit')
        
def id_digit_data(data, rate, tol = 10, prom = 10E6):
    """ Uses find_peaks() to determine frequencies at local maxima
        Matches these frequencies to a digit by looping through nums dictionary of
        tone frequencies to find closest ones given a tolerance
        
        Input: data array (samples), rate (samples/sec), tolerance (default = 10),
        prominence of peaks we seek (default = 10E6)
        Return: Matched digit (as a string) or x if no digit found
    """
    tones, nums, pairs = tone_data()
    transform = fft.fft(data)
    freq = fft.fftfreq(data.shape[0], 1/rate)
    freq = fft.fftshift(freq)
    transform = fft.fftshift(transform)
    abs_trans = np.abs(transform)    
    plt.loglog(freq, abs_trans, '.k')
    pks_loc = find_peaks(abs_trans, prominence=prom)
    pks_loc = pks_loc[0][2:4] # select out last two, since first two are negative versions
    freq1 = freq[pks_loc[0]] # lower freq
    freq2 = freq[pks_loc[1]] # higher freq
    
    prevErr = float('inf')  # just to start off the lastErr holder var
    digit = -1
    # loops through all of the frequency pairs for each digit and
    # compares to pair from transform. Uses 2norm to find error between the two
    # and the smallest error wins out
    # this gives some tolerance to the code, so that even if discreet measured
    # freq don't land on any of the tone frequencies exactly, it will still match
    # them correctly
    for i in nums:
        currErr = \
            np.sqrt((nums[i][0] - freq1)**2 + (nums[i][1] - freq2)**2)
        if currErr < prevErr and currErr < tol:
            digit = i
            prevErr = currErr
 
    if digit != -1: # if two frequencies were not found
        print(f'Digit found: {digit}')
        return str(digit) # grabs digit corresponding to freq pair
    else:
        print('no frequency pair to match')
        return 'x'

def identify_dial(fname, prominence = 10E6):
    """ Breaks up a sound file, given assumed tone length of 0.7, into individual
        tones and calls id_digit_data() with their data and sound file sampling rate
        
        Input: wav file name, prominence of peaks we seek (default = 10E6)
        Return: dialed number as a string: xxx-xxxx
    """
    tone_length = 0.7  # signal broken into 0.7 sec chunks with one num each
    rate, data, sound_length = load_wav(fname)
    tones, nums, pairs = tone_data()
    chunks = int(sound_length/tone_length)
    len_chunks = len(data)//chunks
    digits = ""
    # for each chunk, identify the digit    
    split_dial = [data[i:i + len_chunks] for i in range(0, len(data), len_chunks)]
    for k in range(chunks):
        # digits[k] = id_digit_data(split_dial[k], rate)
        digits += id_digit_data(split_dial[k], rate, prom = prominence)
        if k == 2:
            digits += "-"
    # then print the number dialed at the end
    print(digits)

    
if __name__ == "__main__":
    # freq, transf = identify_digit('5.wav')
    # imaginary = np.imag(transf)
    # reals = np.real(transf)
    # maxloc = np.abs(imaginary).argmax()
    
    # Question 1a
    freq, F = dft('0.wav') 
    # k/L checked within function -> prints True if conversions correct
    
    print('--------------------------')
    # freq, abs_F = dft_mag('0.wav')
    # # frequency list contains all of the frequencies, so now just need the freqs @ peak
    # pks_loc = find_peaks(abs_F, prominence=10E6)
    # pks_loc = pks_loc[0][0:2] # select out first two, since second two are positive versions
    # # of same frequency
    # freq1 = int(round(abs(freq[pks_loc[0]])))
    # freq2 = int(round(abs(freq[pks_loc[1]])))
    # # code tested and working for both 0 and 5 wav, move to identify_digit function
    
    # Question 1aa
    digit = identify_digit('5.wav')
    # successfully prints 5
    print('--------------------------')
    # Question 1b
    identify_dial('dial.wav', prominence=10E6)
    identify_dial('dial2.wav', prominence=10E6)
    # DIAL1: 555-3429
    # DIAL2: 800-6284
    
    # Question 1c
    identify_dial('noisy_dial.wav', prominence=17E5)
    # NOISY DIAL: 555-3429
    