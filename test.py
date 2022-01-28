from itertools import combinations
from os import pipe


import numpy as np
import pandas as pd
import scipy as sp
import logging
import datetime
import sys
import glob

import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

def promotion_strategy(df,threshold):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    classifier=pickle.load(open(glob.glob('*.pkl')[0],'rb'))
    
    
    y_score = classifier.predict_proba(df)[:,1]
    #set threshold to be 0.373015873015873 which is determined from the starbuck file
    
    #threshold = 0.37461969618746094 #(random forest)
    y_pre = np.zeros((len(y_score),1))
    y_pre[y_score>threshold]=1
    
    my_dict={1:'Yes',0:'No'}
    promotion=np.vectorize(my_dict.get)(y_pre)
    return promotion

def score(df, promo_pred_col = 'Promotion'):
    n_treat       = df.loc[df[promo_pred_col] == 'Yes',:].shape[0]
    n_control     = df.loc[df[promo_pred_col] == 'No',:].shape[0]
    n_treat_purch = df.loc[df[promo_pred_col] == 'Yes', 'purchase'].sum()
    n_ctrl_purch  = df.loc[df[promo_pred_col] == 'No', 'purchase'].sum()
    irr = n_treat_purch / n_treat - n_ctrl_purch / n_control
    nir = 10 * n_treat_purch - 0.15 * n_treat - 10 * n_ctrl_purch
    return (irr, nir)
    

def plot_irr_nir(promotion_strategy,threshold_iter):
    '''plot the irr and nir against threshold

    args:
        promotion_strategy(callable): a function that predict whether a user might purchase
        threshold_iter(iterables): an iterables containing thresholds
    '''

    test_data = pd.read_csv('data/Test.csv')
    df = test_data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]

    # promos=promotion_strategy(df)
    # score_df = test_data.iloc[np.where(promos.flatten() == 'Yes')[0]]    
    # irr, nir = score(score_df)

    thre_irr_nir = []

    for threshold in threshold_iter:
        promos=promotion_strategy(df,threshold)
        score_df = test_data.iloc[np.where(promos.flatten() == 'Yes')[0]]    
        irr, nir = score(score_df)
        thre_irr_nir.append((threshold,irr,nir))
    
    thre_list, irr_list, nir_list = zip(*thre_irr_nir)

    plt.figure()
    ax1=plt.subplot(211)
    ax1.plot(thre_list,irr_list)
    ax1.set_xlabel('decision threshod')
    ax1.set_ylabel('Incremental response rate')

    ax2=plt.subplot(212)
    ax2.plot(thre_list,nir_list)
    ax2.set_xlabel('decision threshod')
    ax2.set_ylabel('Net incremental revenue')

    plt.show()

    # print("Nice job!  See how well your strategy worked on our test data below!")
    # print('Your irr with this strategy is {:0.2f}.'.format(irr))
    # print()
    # print('Your nir with this strategy is {:0.2f}.'.format(nir))
    
    # print("Approximately, the highest scores obtained were: irr of {} and an nir of {}.\n\n How did you do?".format(0.1, 300))
    # return irr, nir

if __name__ == '__main__':
    plot_irr_nir(promotion_strategy,threshold_iter=np.linspace(0.2,0.3,num=500))
    