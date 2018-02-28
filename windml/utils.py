#%%############################################################################ imports

import copy
import inspect
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy as sp
import scipy.ndimage
import shutil
import sklearn
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.externals
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import sklearn.utils
#-----------------------------------------------------
import skimage
#_____________________________________________________

#%%############################################################################# definitions
import time

from statsmodels.emplike.koul_and_mc import params

from windml.example import Y1


def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)



def class_quality(real,pred):
    jac=sklearn.metrics.jaccard_similarity_score(real,pred)
    acc=sklearn.metrics.accuracy_score(real,pred)
    ham=1.0-sklearn.metrics.hamming_loss(real,pred)
    pre=sklearn.metrics.precision_score(real,pred,average='weighted')
    rec=sklearn.metrics.recall_score(real,pred,average='weighted')
    fis=sklearn.metrics.f1_score(real,pred,average='weighted')
    cor=sp.stats.pearsonr(real,pred)[0]
    return jac,acc,ham,pre,rec,fis,cor

def call_with_kwargs(func,kwargs,para=None): # kwargs={**kwargs,**kwargz}
    params=inspect.getargspec(func).args
    dicts={k: v for k,v in kwargs.items() if k in params}
    if(para==None):
        f=func(**dicts)
    else:
        f=func(*para,**dicts)
    return f

def downsample_image(XX,downs=1,speed=1):
    X=np.vstack([XX[:,:speed].T,skimage.transform.resize(XX[:,speed:],(len(XX),int((len(XX[0])-speed)/downs))).T]).T
    return X

def stripe_image(X,Y,breadth=1):
    lx=len(Y)
    x=[]
    y=Y[0:lx-breadth+1]
    for i in range(lx-breadth+1):
        x.append(X[i:i+breadth].ravel())
    x=np.array(x)
    return x,y

def binarize_image(YY, multi=1, Y=None):
    if(multi):
        Y=YY
    else:
        Y=1*(Y>0)
    return Y

def grid_search(XX,YY,params,order,n_jobs=2,test_size=0.2,clf=sklearn.svm.SVC,metric=sklearn.metrics.jaccard_similarity_score):

    keys=list(params.keys())
    vals=list(params.values())
    values=[{keys[ji]:j for ji,j in enumerate(k)} for k in np.itertools.product(*vals)]
    idcs=[range(len(v)) for v in vals]
    indics=[{keys[ji]:j for ji,j in enumerate(k)} for k in np.itertools.product(*idcs)]
    V=len(values)

    quali=np.zeros([len(params[o]) for o in order])*np.nan
    qualt=np.zeros([len(params[o]) for o in order])*np.nan

    path = tempfile.mkdtemp()
    memoi = os.path.join(path,'memoi.mmap')
    memot = os.path.join(path,'memot.mmap')
    quali = np.memmap(memoi, dtype=quali.dtype, shape=quali.shape, mode='w+')
    qualt = np.memmap(memot, dtype=qualt.dtype, shape=qualt.shape, mode='w+')

    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(simple_search)(XX,YY,v,clf,order,indics,test_size,metric,vi,V,quali,qualt) for vi,v in enumerate(values))

    try:
        shutil.rmtree(path)
    except:
        pass

    mu=np.nanmean(quali,0)
    sd=np.nanstd(quali,0)
    idm=np.unravel_index(np.nanargmax(mu),mu.shape)
    valui={o:params[o][idm[oi]] for oi,o in enumerate(order[1:])}

    mu=np.nanmean(qualt,0)
    sd=np.nanstd(qualt,0)
    idm=np.unravel_index(np.nanargmax(mu),mu.shape)
    valut={o:params[o][idm[oi]] for oi,o in enumerate(order[1:])}

    X=downsample_image(XX,downs=valui['downs'],speed=valui['speed'])
    X,Y=stripe_image(X,YY,breadth=valui['brdth'])
    Y=binarize_image(Y,multi=valui['multi'])

    return X,Y,quali,valui,qualt,valut

def simple_search(XX,YY,value,clf,order,indics,test_size,metric,vi,V,quali,qualt):

    dd=value['downs']
    ii=value['iters']
    ss=value['speed']
    mm=value['multi']
    bb=value['brdth']

    X=downsample_image(XX,downs=dd,speed=ss)
    X,Y=stripe_image(X,YY,breadth=bb)
    Y=binarize_image(Y,multi=mm)

    fnc=call_with_kwargs(clf,value)
    idv=indics[vi]
    idx=[idv[o] for o in order]

    temp=time.time()
    Y0=[0]
    while(set(Y)!=set(Y0)):
        X0,X1,I0,I1=sklearn.model_selection.train_test_split(X,np.array(range(len(Y))),train_size=value['train_size'],test_size=test_size)
        Y0,Y1=Y[I0],Y[I1]
    fnc.fit(X0,Y0)
    y1=fnc.predict(X1)
    dura=(time.time()-temp)/60.0
    quali[tuple(idx)]=metric(Y1,y1)
    qualt[tuple(idx)]=dura
    print(vi,V,dura,X.shape,value)

    return None

def sliding_ensemble_quality(Y1,y1,window=3,weighted=1):
    labs=sorted(list(set(Y1)))
    freq=np.array([(Y1==l).sum() for l in labs])
    freq=1.0*freq/np.nansum(freq)
    z=Y1.copy()
    if(len(y1.shape)==1):
        yy=np.array([y1])
    else:
        yy=y1
    lx=len(y1)
    for ii,i in enumerate(y1):
        xmi,xma=max(0,ii-window),min(lx,ii+window+1)
        freqs=np.array([(yy[:,xmi:xma]==l).sum() for l in labs])
        freqs=1.0*freqs/np.nansum(freqs)
        if(weighted):
            ratio=1.0*freqs/freq                                                    # weighted majority voting
        else:
            ratio=freqs
        ratio[~np.isfinite(ratio)]=0
        z[ii]=np.nanargmax(ratio)
    return z

def window_search(X,Y,value,windows,test_size=0.2,clf=sklearn.svm.SVC,metric=sklearn.metrics.jaccard_similarity_score):
    Y0=[0]
    while(set(Y)!=set(Y0)):
        X0,X1,I0,I1=sklearn.model_selection.train_test_split(X,np.array(range(len(Y))),train_size=value['train_size'],test_size=test_size)
        Y0,Y1=Y[I0],Y[I1]
    fnc=call_with_kwargs(clf,value)
    fnc.fit(X0,Y0)
    y=fnc.predict(X)
    qs=np.zeros(len(windows))
    for wi,w in enumerate(windows):
        z=sliding_ensemble_quality(Y,y,window=w)
        qs[wi]=metric(z,Y)
    window=windows[np.nanargmax(qs)]
    z=sliding_ensemble_quality(Y,y,window=window)
    qy0=metric(y[I0],Y[I0])
    qy1=metric(y[I1],Y[I1])
    qz0=metric(z[I0],Y[I0])
    qz1=metric(z[I1],Y[I1])
    copt=copy.deepcopy(fnc)
    return copt,qs,window,y,z,qy0,qy1,qz0,qz1,I0,I1

def random_quality(Y,R,metric=sklearn.metrics.jaccard_similarity_score):
    y=Y.copy()
    q=[]
    for r in range(R):
        random.shuffle(y)
        q.append(metric(y,Y))
    return np.array(q)

def plot_multi_label(ss,Y,offset,width,label=0,alpha=0.8):
    S=len(ss)
    ll=range(len(Y))
    for si,s in enumerate(ss):
        color=plt.cm.rainbow(si/(S-1.0))
        if(label):
            plt.fill_between(ll,width*(si+offset),width*(si+offset+1),where=(Y==s),color=color,alpha=alpha,label=ss[si],lw=0)
        else:
            plt.fill_between(ll,width*(si+offset),width*(si+offset+1),where=(Y==s),color=color,alpha=alpha,lw=0)
    return None

def plot_overview(X,Y,quali,value,windows,labels,order,test_size=0.2,ylabel='quality',clf=sklearn.svm.SVC,metric=sklearn.metrics.jaccard_similarity_score):

    mu=np.nanmean(quali,0)
    sd=np.nanstd(quali,0)
    idm=np.unravel_index(np.nanargmax(mu),mu.shape)


    P=max(len([i for i in mu.shape if i>1]),5)
    L=len(labels.keys())
    lx,ly=X.shape
    T=int(lx*value['train_size'])
    plt.clf()
    count=-1
    for ji,j in enumerate(mu.shape):
        if(j==1):
            continue
        count+=1
        plt.subplot(2,P,1+count)
        idj=[i for i in idm]
        idj[ji]=slice(None)
        mux=mu[tuple(idj)]
        sdx=sd[tuple(idj)]
        o=order[1:][ji]
        xx=params[o]
        plt.title(xx[idm[ji]])
        if(type(xx[0])==str or type(xx[0])==tuple):
            xx=range(len(xx))
        plt.fill_between(xx,mux-sdx,mux+sdx,lw=0,color='LightGray')
        plt.plot(xx,mux,color='black',lw=2)
        plt.scatter(xx[idm[ji]],mux[idm[ji]],color='purple',marker='o',zorder=9)
        plt.xlabel(o)
        plt.ylabel(ylabel)
        if(o=='C' or o=='gamma' or o=='alpha'):
            plt.xscale('log')

    copt,qs,window,y,z,qy0,qy1,qz0,qz1,I0,I1=window_search(X,Y,value,windows,clf=clf,metric=metric,test_size=test_size)

    qr=random_quality(Y1,200,metric=metric)

    plt.subplot(2,P,2*P-4)
    plt.title(window)
    plt.plot(windows,qs,lw=2,color='black')
    plt.scatter(window,max(qs),marker='o',color='purple',zorder=9)
    plt.xlabel('window')
    plt.ylabel('quality')

    plt.subplot(2,P,2*P-3)
    plt.title('%.3f'%np.nanmean(qr))
    hh=plt.hist(qr,bins=30,histtype='step',lw=2,color='gray',normed=1)
    plt.plot([qy0]*2,[0,max(hh[0])*0.9],lw=2,color='black',ls='--')
    plt.plot([qz0]*2,[0,max(hh[0])*0.9],lw=2,color='black')
    plt.plot([qy1]*2,[0,max(hh[0])*0.8],lw=2,color='purple',ls='--')
    plt.plot([qz1]*2,[0,max(hh[0])*0.8],lw=2,color='purple')
    plt.xlabel(ylabel)
    plt.ylabel('probability')

    plt.subplot(2,P,2*P-2)
    plt.title('train '+'%.3f'%qy0+' '+'%.3f'%qz0+'\n test '+'%.3f'%qy1+' '+'%.3f'%qz1)
    plt.imshow(X.T,origin='upper',interpolation='nearest',cmap='gray',aspect='auto')
    plot_multi_label(labels,Y,(L+2)*0,ly*0.02,alpha=0.6,label=1)
    plot_multi_label(labels,y,(L+2)*1,ly*0.02,alpha=0.6,label=0)
    plot_multi_label(labels,z,(L+2)*2,ly*0.02,alpha=0.6,label=0)
    plt.plot([lx*T]*2,[0,ly],lw=2,color='white',ls='-')
    plt.legend(loc=4)
    plt.axis([0,lx,ly,0])
    plt.xlabel('time')
    plt.ylabel('frequency')

    plt.subplot(2,P,2*P-1)
    plt.title('train confusion matrix')
    mat=sklearn.metrics.confusion_matrix(Y[I0],z[I0])
    plt.imshow(mat,origin='upper',interpolation='nearest',cmap='gray',aspect='auto')
    plt.xlabel('real')
    plt.ylabel('pred')

    plt.subplot(2,P,2*P-0)
    plt.title('test confusion matrix')
    mat=sklearn.metrics.confusion_matrix(Y[I1],z[I1])
    plt.imshow(mat,origin='upper',interpolation='nearest',cmap='gray',aspect='auto')
    plt.xlabel('real')
    plt.ylabel('pred')

    return copt,window,value

def select_clf(ki,base):
    if(ki==0):
        clf=sklearn.tree.DecisionTreeClassifier
        order=base+['max_features','max_depth']
    elif(ki==1):
        clf=sklearn.naive_bayes.GaussianNB
        order=base+[]
    elif(ki==2):
        clf=sklearn.neighbors.KNeighborsClassifier
        order=base+['n_neighbors','weights']
    elif(ki==3):
        clf=sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        order=base+['solver','shrinkage']
    elif(ki==4):
        clf=sklearn.neural_network.MLPClassifier
        order=base+['alpha','hidden_layer_sizes']
    elif(ki==5):
        clf=sklearn.svm.SVC
        order=base+['cache_size','kernel','C','gamma']
    return clf,order