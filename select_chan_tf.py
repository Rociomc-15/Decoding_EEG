# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:48:39 2022

SCRIPT DECODING TIME-FREQUENCY EEG
@author: Rocio Martinez Caballero

    rociomartinez@ugr.es 
    
"""


"""
1)IMPORT PACKAGES 
"""

#!/usr/bin/env python3

import sys
import scipy.io as sio
import os
import numpy as np
import pandas as pd #carga de datos

from numpy import  newaxis

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
#from jr.plot import plot_butterfly, plot_gfp, pretty_colorbar

import matplotlib
import random
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import LeaveOneOut

# Function AUC (area under the curve)
def scorer(y_true, y_pred):
    """Proxy for effect size estimate, should it be uni- or multivariate"""
    #y_pred = y_pred[:, 1] if y_pred.ndim == 2 else y_pred # ensure dimensionality
    return roc_auc_score(y_true, y_pred)



chan = int(sys.argv[1]) #channels analysis
n_freq  =int(sys.argv[2]) #point freq
print("analysis time-freq. Channel: ", chan)


"""
1)IMPORT X DATA 4D (trials x time x freq x channels))
"""
directory = 'data'

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    os.chdir(path)
    #PASO 1 - Importar datos X e Y
    files = os.listdir('tf_features3')
    files = sorted(files)
   
    #se aletorizan los datos
    np.random.shuffle(files)
   
    files1 = os.listdir('preprocessed')
    os.chdir('tf_features3')
   
    #Se cargan los datos aletorizados en X
    mat = sio.loadmat(files[0])
    a= mat.get('X')
    a = np.delete(a,  [0,1,2,78,79,80], axis=0)
    a = a[newaxis, :, :,:]
   
    for trial in files:
      mat1 = sio.loadmat(trial)
      b = mat1.get('X')
      b = np.delete(b, [0,1,2,78,79,80], axis=0)
      b = b[newaxis, :, :,:]
      a = np.row_stack((a,b))
     
   
    X = a[1:]
   
    os.chdir('..')
    dataset = pd.read_csv('labels.csv')
   
   
    """
    3) IMPORT Y_LABELS 
    """

   
    Y = dataset.iloc[:,1].values
   
    Y1=[]
   
    #importamos el orden de los ensayos de X para copiarlos en Y
   
    import re
    idx=([float(s) for s in re.findall(r'-?\d+\.?\d*', "".join(files))])
   
    #pasamos la lista a tipo int
    for m in range(len(idx)):
        Y1.append(int(idx[m])-1)
   
    #asignamos las etiquetas correspondientes a cada ensayo
    Y2=[]
    for j in Y1:
        Y2.append(Y[j])
    
    
    #Codificamos los valores de Y2 en left(0) Y right(1)
    for i in range(len(Y2)):
           if (( Y2[i] == 457 ) or ( Y2[i]  == 456)):
                 Y2[i] = 1
           else:
                Y2[i] = 0
    """           
    #Codificamos los valores de y en hit o ilu
     
    for i in range(len(Y)):
        if ( Y[i] == 457 ) or ( Y[i] == 557 ):
                Y[i] = 0
        
        else:
                Y[i] = 1
    """
    
    #Replicamos las etiquetas para cada ensayo        
    y=np.array(Y2).reshape(-1,1)
   
   
    os.chdir('..')
    os.chdir('..')

   
    print("Import data and labels subject: ", filename)
    
    """
    4) CLASSIFIER 
    """
    # Let's define a linear SVM classifier. To optimize the result, we use
    # i) a normalization step,
    # ii) a probabilistic output.
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    # As the output is continuous and not discrete, the scoring metrics is an Area Under the Curve.
    # Note: scikit-learn doesn't score probabilistic estimates by default, so we're building it ourself.

    # To avoid over-fitting we have to use a cross-validation procedure. Since it is
    # a classification problem, it's better to use stratified folding
    #cv = StratifiedKFold()
    # Let's train and test a classifier at each time sample
    scores = []
    
    """
    5) CROSS VALIDATION LEAVE ONE OUT
    """
    loo = LeaveOneOut()
    loo.get_n_splits(X,y)
   
   
    n_time =len(X[1])
    n_trial = len(idx)
    score = np.zeros((n_freq, n_time))
    subjects_score = []

   
    for time in range(n_time):
        for freq in range(n_freq):
            score_cv = []
            y_pred = np.zeros((n_trial, 2)) # Probabilistic estimates are reported for each class, hence n = 2
            for train, test in loo.split(X=X, y=y):  
                # Fit on train set
                clf.fit(X[train, time,freq,chan].reshape(-1, 1), y[train].ravel())
                score[freq, time] = scorer(y, X[:, time, freq,chan].reshape(-1))
               
   
    subjects_score.append(score)   
    sio.savemat('subjects_concatenate' + str(chan) + '.mat', {'subjects_score': subjects_score})
    print("Finish the subject: ", filename)




fig,ax1 = plt.subplots()
cont1 = ax1.contourf(np.mean(subjects_score,axis=0),cmap='rainbow',extent=[20,3900,5,119])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Frequences (Hz)')
ax1.set_title('AUC in the channel x')
plt.colorbar(cont1,ax=ax1)
plt.yticks()
plt.show()

