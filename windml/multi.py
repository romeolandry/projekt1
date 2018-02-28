#%%############################################################################ imports

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import sys
import time

home=os.path.expanduser("~/")
os.chdir(home)

#%%############################################################################ test functions

def func_sleep(T):
    time.sleep(1.0/T) # 10
    return None

def func_matrix(T):
    for k in range(int(100/T)): # 1000
        a=sp.rand(100,100)
        a=a**5
    return None

#%%############################################################################ function for testing speed gain due to parallelization

def multi():
    
    #%%# parameters
    
    F=2 # different functions
    C=4 # number of cores
    I=3 # iterations
    T=1 # chunk number # 1/100 is good/bad for parallelization

    functions=[func_sleep,func_matrix]
    
    #%%# auxiliary variables
    
    FF=range(F)
    CC=range(1,C+1)
    II=range(I)
    TT=range(T)

    duration=np.zeros((F,C,I))*np.nan

    #%%# loop over parameters and functions

    for fi,f in enumerate(FF):
        for ci,c in enumerate(CC):
            for ii,i in enumerate(II):
                print(fi,F,ci,C,ii,I)
                func=functions[fi]
                temp=time.time()
                joblib.Parallel(n_jobs=c)(joblib.delayed(func)(T) for t in TT)
                duration[fi,ci,ii]=time.time()-temp

    #%%# plot computation times

    for fi,f in enumerate(FF):

        color='C'+str(fi)
        label=str(functions[fi].__name__)

        mu=np.nanmean(1.0/duration[fi],-1)
        sd=np.nanstd(1.0/duration[fi],-1)
        ln=mu[0]*np.array(CC)

        plt.fill_between(CC,mu-sd,mu+sd,lw=0,color=color,alpha=0.5)
        plt.plot(CC,mu,lw=2,ls='-',color=color,label=label)
        plt.plot(CC,ln,lw=2,ls='--',color=color)

    plt.legend()
    plt.xticks(CC,CC)
    plt.ylim(ymin=0)
    plt.xlabel('cores')
    plt.ylabel('calls per second')
    plt.savefig('multi.png')

#%%############################################################################ run script with dummy arguments

args=sys.argv
print(args)
multi()

