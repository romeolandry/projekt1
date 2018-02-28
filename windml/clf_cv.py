#%%############################################################################ imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import scipy.ndimage
import sklearn.utils
import  collections
import pydotplus

from sklearn import tree, grid_search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sphinx.ext import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

home=os.path.expanduser("~/")
os.chdir(home)

#%%############################################################################# Decision tree

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
#plt.show()


#%%# set metric for measuring classification quality

metric = sklearn.metrics.jaccard_similarity_score

# set size/fraction of training and test sets
train_size = 0.5
test_size = 0.2

# split data into training (0) and test set (1)
X0,X1,Y0,Y1 = train_test_split(XX, YY, train_size=train_size, test_size=test_size)


#cross validation with decision Tree

# parameters = {'criterion':'gini', 'random_state':100,'max_deph':range(3,20), 'min_samples_leaf':5 }
# clf = GridSearchCV(tree.DecisionTreeClassifier(),parameters, n_jobs=4)
# clf.fit(X0, Y0)
# tree_model = clf.best_estimator_
# print(clf.best_score_,clf.best_params_)

depth = []
for i in range(3, 20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    # perform 7-fold cross validation
    scores = cross_val_score(estimator=clf,X= XX, y=YY, cv=2, n_jobs=2)
    clf.fit(X0,Y0)
    depth.append(scores.mean())
print(depth)

# predict classes for complete data
yy = clf.predict(XX)

# prediction quality
q = metric(yy, YY)

#%%# plot
plt.plot(YY, ls='', marker='o', color='red', label='manual')
plt.plot(yy+0.1, ls='', marker='o', color='orange', label='prediction') # shift for better visibility

plt.title('quality = '+'%.2f'%q)
plt.legend(loc=0)
plt.ylabel('class')
plt.yticks(list(labels.keys()),list(labels.values()))
plt.xlim(0,lx)
plt.show()
