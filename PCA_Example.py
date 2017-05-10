# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:39:08 2017

@author: pmacias
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)


print(pca.explained_variance_ratio_) 
