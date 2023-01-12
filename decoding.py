# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:11:12 2021

SCRIPT DECODING EEG
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
import pandas as pd #datas
import statistics
from numpy import  newaxis

import matplotlib
import random
import plotly.graph_objects as go
import numpy as np


import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score


# Function AUC (area under the curve)
def scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])





n_time = int(sys.argv[1]) #points time
"""
1)IMPORT X DATA 3D (trials x time X (channels x freq))
"""
y_p = []
scor = []
directory = 'data'
for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        os.chdir(path)
        #files = os.path.join(directory, filename,'tf_features')    
        #PASO 1 - Importar datos X e Y
        files = os.listdir('tf_features')
        #files = os.listdir(files)
        files = sorted(files)
   
        #se aletorizan los datos
        np.random.shuffle(files)
            
        files1 = os.listdir('preprocessed')
        os.chdir('tf_features')
        #files1 = os.path.join(directory, filename,'preprocessed')     
        #files1 = os.listdir(files1)
        
        #Se cargan los datos aletorizados en X
        mat = sio.loadmat(files[0])
        a= mat.get('X')
        a = np.delete(a,  [0,1,2,78,79,80], axis=0)
        a = a[newaxis, :, :]
    
        for trial in files:
            mat1 = sio.loadmat(trial)
            b = mat1.get('X')
            b = np.delete(b, [0,1,2,78,79,80], axis=0)
            b = b[newaxis, :, :]
            a = np.row_stack((a,b))
        
    
        X = a[1:]
    
    
        """
        3) IMPORT Y_LABELS 
        """

        os.chdir('..')
        dataset = pd.read_csv('labels.csv')
    
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
    
        #Codificamos los valores de Y2 en RIGHT(0) Y LEFT(1)
        for i in range(len(Y2)):
            if (( Y2[i] == 457 ) or ( Y2[i]  == 456)):
                    Y2[i] = 1
            else:
                    Y2[i] = 0
    
        #Replicamos las etiquetas para cada ensayo         
    
        y=np.array(Y2).reshape(-1,1)
        

        os.chdir('..')
        os.chdir('..')
    
        """
        4) CLASSIFIER 
        """
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

        """
        5) CROSS VALIDATION K=5(DEFAULT)) 
        """
        cv = StratifiedKFold()
        scores = []
    

        n_trial = len(idx)
    
        for time in range(n_time):
            score_cv = []
            y_pred = np.zeros((n_trial, 2)) # Probabilistic estimates are reported for each class, hence n = 2
            for train, test in cv.split(X=X, y=y):
                # Apply the clasiffier to X_train
                clf.fit(X[train, time,:], y[train].ravel())
                # Predict the features
                y_pred[test, :] = clf.predict_proba(X[test, time, :])
                
            score = scorer(y, y_pred) #AUC score
            scores.append(score)
        
        
        if (scor == []):
            scor = scores
        else:
            scor_mean = np.mean([scor,scores], axis = 0) #avergae
            scor_std =  np.std([scor,scores], axis = 0) #std
        
        print("Finish the subject: ", filename)
        
        

"""
6)Plot AUC-time (averge subjects), Figure.
"""

#random color generation in plotly
hex_colors_dic = {}
rgb_colors_dic = {}
hex_colors_only = []
for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)
    hex_colors_dic[name] = hex
    rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

#c = random.choice(hex_colors_only)
c = '#7B68EE'


fig = go.Figure()


fig.add_trace(go.Scatter( x=times, y=scor_mean+scor_std,
                                     mode='lines',
                                     line=dict(color=c,width =0.1),
                                     name='upper bound'))


fig.add_trace(go.Scatter(x=times,y= scor_mean,
                         mode='lines',
                         line=dict(color=c),
                         fill='tonexty',
                         name='mean'))

fig.add_trace(go.Scatter(x=times,y= scor_mean-scor_std,
                         mode='lines',
                         line=dict(color=c, width =0.1),
                         fill='tonexty',
                         name='lower bound'))

"""
fig.add_trace(go.Scatter(x = [2000,2000], y = [0.2, 0.9],  
                         line = dict(color='black', width =1.5,dash='dash'),
                         name='Instruction'))
"""

fig.update_layout(
    xaxis_title="Time(s)",
    yaxis_title="AUC",
    title='AUC-time'
)

   
fig.show()
fig.write_html('deconding', auto_open=True)
