#%%############################################################################ imports

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage
import sklearn.utils
from cv_funktion import run_gridsearch, best_alogrithmus
from sklearn import svm
from sklearn.tree import tree

home=os.path.expanduser("~/")
os.chdir(home)

#%%############################################################################# cross-validation

# select path and file
path = '/home/romeo/Nextcloud/RI-Romeo/WindML/Data/'
file = 'Wav1'

# read and rescale image / spectrogram
XX = sp.ndimage.imread(path+file+'.png').T/255.0
ly,lx= XX.shape

# read manual classification data
YY = np.array(pd.read_csv(path+file+'.txt')['Status'])

# set labels of different noises
labels = {1:'normal', 2:'other', 3:'vehicles', 4:'voices', 5:'setup', 6:'airplanes'}

#%%# plot data

plt.subplot(211)
plt.imshow(XX.T, cmap='Greys', interpolation='nearest', aspect='auto', extent=[0,lx,0,ly])
plt.xlabel('time')
plt.ylabel('frequency')

plt.subplot(212)
plt.plot(YY, ls='', marker='o', color='red', label='manual')
plt.legend(loc=0)
plt.ylabel('class')
plt.yticks(list(labels.keys()),list(labels.values()))
plt.xlim(0,lx)


#%%# set metric for measuring classification quality

metric = sklearn.metrics.jaccard_similarity_score

# set size/fraction of training and test sets
train_size = 0.5
test_size = 0.2

# split data into training (0) and test set (1)
X0,X1,Y0,Y1 = sklearn.model_selection.train_test_split(XX, YY, train_size=train_size, test_size=test_size)

# svc = sklearn.svm.SVC(gamma=1.0, C=1.0).fit(X0, Y0)

param_grid_dt = {"criterion":["gini"],
                 "min_samples_split": [2, 10, 20],
                 "max_depth": [None, 2, 5, 10],
                 "min_samples_leaf": [1, 5, 10],
                 "max_leaf_nodes": [None, 5, 10, 20],
                 }
Cs = Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_gird_svm = {'C': Cs, 'gamma' : gammas}


print ("-- Grid Parameter Search via 2-fold CV")

dt= tree.DecisionTreeClassifier()
#dt_gs = run_gridsearch(XX, YY, dt, param_gird_dt, cv=2)
dt_gs = best_alogrithmus(dt, svm, X0,Y0, param_grid_dt,param_gird_svm,cv=2)


# clf_gini = sklearn.tree.DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=i, min_samples_leaf=5)
#
# param1 = range(2,30)
# param2 = [dct, svn, cvn]
