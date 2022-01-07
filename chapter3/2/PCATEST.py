# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:43:40 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2], 
                  random_state =9)
fig = plt.figure(0)
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

plt.figure(1)
X_new = pca.transform(X)
plt.plot(X_new[:, 0], X_new[:, 1],'o')