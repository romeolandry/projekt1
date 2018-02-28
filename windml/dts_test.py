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

from sklearn import tree
from sklearn.model_selection import train_test_split
from sphinx.ext import graphviz
from sklearn.cross_validation import cross_val_score

home=os.path.expanduser("~/")
os.chdir(home)

#%%############################################################################# Decision tree

# # select path and file
# path = '/home/romeo/Nextcloud/RI-Romeo/WindML/Data/'
# file = 'Wav6'
#
# # read and rescale image / spectrogram
# XX = sp.ndimage.imread(path+file+'.png').T/255.0
# ly,lx= XX.shape
#
# # read manual classification data
# YY = np.array(pd.read_csv(path+file+'.txt')['Status'])

# select path and file
path = '/home/romeo/Dokumente/projekt1_maschinelles_lernen/windml/data/image/'
path2 ='/home/romeo/Dokumente/projekt1_maschinelles_lernen/windml/data/label/'
file = 'Wav'


# read and rescale image / spectrogram
listing = os.listdir(path)
listing_label = os.listdir(path2)
num_samples= np.size(listing)
print(num_samples)

for file  in listing :
    XX = plt.imread (path+file).T/255.0
    ly,lx = XX.shape


for file1 in listing_label :
    YY = np.array(pd.read_csv(path2 + file1)['Status'])


data = (XX,YY)


immatrix = np.array([np.array(XX)]).flatten()


# set labels of different noises
labels = {1:'normal', 2:'other', 3:'vehicles', 4:'voices', 5:'setup', 6:'airplanes'}


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


# multilayer  perceptron
# theano + lasagne
# train SVM, may take a few seconds
# Anwendung von Decision zur Training Dateien

# Anpassung v von Dateien zur proben
# {'criterion': 'gini', 'min_samples_split': 10, 'max_leaf_nodes': 20, 'max_depth': None, 'min_samples_leaf': 10}
depth = []
for i in range(3, 20):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini', random_state=100, max_depth= None, min_samples_leaf=10, min_samples_split=10, )
    # perform 7-fold cross validation
    scores = cross_val_score(estimator=clf_gini,X= XX, y=YY, cv=2, n_jobs=2)
    clf_gini.fit(X0,Y0)
    depth.append(scores.mean())
print(depth)
#tree auf dem Bildschirm Anzeigen auf der Internet Seite http://webgraphviz.com/ den TxtDatei ausfüren
with open("/home/romeo/Nextcloud/RI-Romeo/ProjektML/enercon/python/windml/bild.txt", 'w') as f:
    f = tree.export_graphviz(clf_gini, out_file=f)
data_feature_names = ['Time[s]','Rotational_speed[rpm]']
dot_data = tree.export_graphviz(clf_gini,
                                feature_names= None,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
# predict classes for complete data
yy = clf_gini.predict(XX)

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


