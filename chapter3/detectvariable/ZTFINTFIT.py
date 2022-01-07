# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:20:30 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
from scipy.fftpack import fft,ifft,fftshift
#from numpy.fft import fft,ifft,fftfreq
from scipy import signal

def ztfdata(CSV_FILE_PATH, period):
    dfdata = pd.read_csv(CSV_FILE_PATH)
    hjd = dfdata['HJD']
    mag = dfdata['mag']
    rg = dfdata['band'].value_counts()
    try:
        lenr = rg['r']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
     
    nphjd = np.array(hjd)
    npmag = np.array(mag)
    
    try:
        hang = rg['g']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
    
    nphjd = nphjd[hang:]
    npmag = npmag[hang:]
     
    phases = foldAt(nphjd, period)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npmag[sortIndi]

    s1 = np.diff(resultmag,2).std()/np.sqrt(6)
    s2 = np.std(resultmag)
    if (s2/s1)<2:
        return np.array([0,0,0]),np.array([0,0,0])
    if len(resultmag)<50: 
        return np.array([0,0,0]),np.array([0,0,0])

    N = 100
    x = np.linspace(0,1,N)
    y = np.interp(x, phases, resultmag) 

    fft_y = fft(y) 
    mx = np.arange(N)
    half_x = mx[range(int(N/2))]  #取一半区间
    abs_y = np.abs(fft_y) 
    normalization_y = abs_y/N            #归一化处理（双边频谱）                              
    normalization_half_y = normalization_y[range(int(N/2))] 
    normalization_half_y[0] = period/10
    
    return half_x,normalization_half_y


def showdata(CSV_FILE_PATH, period):
    dfdata = pd.read_csv(CSV_FILE_PATH)
    hjd = dfdata['HJD']
    mag = dfdata['mag']
    rg = dfdata['band'].value_counts()
    try:
        lenr = rg['r']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
     
    nphjd = np.array(hjd)
    npmag = np.array(mag)
    
    try:
        hang = rg['g']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
    
    nphjd = nphjd[hang:]
    npmag = npmag[hang:]
    
    phases = foldAt(nphjd, period)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npmag[sortIndi]
    
    return phases, resultmag
#    
filename = '67_2.7794566.csv'    
P = 2.7794566
#filename = '82.csv'    
#P = 0.3751536


phases, resultmag = showdata(filename, P)
resultmag = resultmag-np.mean(resultmag)

sigmaall = np.std(resultmag)
noiseall = np.diff(resultmag,2).std()/np.sqrt(6)
print(sigmaall/noiseall)

N = 100
sx = np.linspace(0,1,N)
sy = np.interp(sx, phases, resultmag)
sy = sy-np.mean(sy)

fft_y = fft(sy) 
abs_y = np.abs(fft_y) 

yinv = ifft(abs_y)


mx = np.arange(N)
half_x = mx[range(int(N/2))]  #取一半区间

normalization_y = abs_y/N           #归一化处理（双边频谱）                              
normalization_half_y = normalization_y[range(int(N/2))]

fs=100
b, a = signal.butter(8, 0.4, 'lowpass')  #2*截止频率/采样频率 2*10/100 =1
w, h = signal.freqs(b, a)
filtedData = signal.filtfilt(b, a, sy)

plt.figure(2)
plt.semilogx(0.5*fs*w/np.pi, 20 * np.log10(abs(h)))
#plt.plot(0.5*fs*w/np.pi, 20 * np.log10(abs(h)))


plt.figure(0)
plt.plot(phases,resultmag,'.',color = 'b')
#plt.plot(sx, sy, '.', color = 'r')
plt.plot(sx, filtedData, color = 'r')
plt.plot(sx[1:], yinv.real[1:], '.', color = 'g')

#plt.plot(sx[1:], np.abs(yinv.real[1:]))
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
plt.title('ZTFJ000009.62+550138.0')   
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.legend(('observed data', 'filtered data', 'ifft data'), loc='upper right')
plt.savefig('ZTFJ000009.62+550138.0.png')



plt.figure(1)
plt.plot(half_x, normalization_half_y)
plt.xlabel('Frequency',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.savefig('FFT.png')