# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:23:04 2021

@author: dingxu
"""

import os,pickle,time
import numpy as np


path = 'Z:\\public\\dingxudata\\'
file = 'savedata01050.txt'
filename = path+file
data = np.loadtxt(filename)

pickle.dump(data,open(path+'savedata01050.pkl','wb'))