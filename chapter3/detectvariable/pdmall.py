# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:00:35 2021

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft
import random


def readfits(fits_file):
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        print(hdulist[0].header['OBJECT'])
        print(hdulist[0].header['RA_OBJ'], hdulist[0].header['DEC_OBJ'])
        
        indexflux = np.argwhere(pdcsap_fluxes > 0)
#        print(sap_fluxes)
        time = tess_bjds[indexflux]
        time = time.flatten()
        flux = pdcsap_fluxes[indexflux]
        flux =  flux.flatten()
        
        return time, flux
    
def computePDM(time, fluxes):
    
    
    S = pyPDM.Scanner(minVal=0.05, maxVal=330, dVal=0.01, mode="frequency")
    P = pyPDM.PyPDM(time, fluxes)
    
    bindata = int(len(time)/10)
        
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    
    return pdmp, delta    
    
    
path = 'J:\\TESSDATA\\section1\\' 
file = 'tess2018206045859-s0001-0000000031655792-0120-s_lc.fits'
tbjd, fluxes = readfits(path+file)
ls = [random.randint(0,len(fluxes)) for i in range(500)]
st = set(ls)
listst = list(st)
nplistst = np.array(listst)

tbjdtemp = []
fluxestemp = []
for i in range (0,len(listst)):
    tbjdtemp.append(tbjd[nplistst[i]])
    fluxestemp.append(fluxes[nplistst[i]])

jd = np.array(tbjdtemp)  
flux = np.array(fluxestemp)  
print('it is ok')   
 
pdmp, delta = computePDM(jd, flux)
plt.figure(0)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('JD',fontsize=18)
plt.ylabel('FLUX',fontsize=18) 

plt.figure(1)
plt.plot(jd, flux, '.')
plt.xlabel('JD',fontsize=18)
plt.ylabel('FLUX',fontsize=18) 


