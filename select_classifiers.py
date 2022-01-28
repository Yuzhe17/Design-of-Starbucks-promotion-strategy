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
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

def label_if_promote(df):
    '''label the if_promote column as 1 if the the promotion is sent and purchase is made,
        label the if_promote column as -1 if the the promotion is not sent and purchase is not made
    '''

    if df['Promotion']=='Yes' and df['purchase']==1:
        df['if_promote']=1

    if df['Promotion']=='No' and df['purchase']==0:
        df['if_promote']=-1
    
def preprocess_data(df):
    '''preprocess the df dataset to add if_promote column indicating if we would send offers to these kind of users.
    '''
    df['if_promote']=np.zeros(df.shape[0])
    
    df.loc[(df['Promotion']=='Yes') & (df['purchase']==1),'if_promote']=1
    df.loc[(df['Promotion']=='No') & (df['purchase']==0),'if_promote']=-1


    return df

def prepare_dataset():
    '''read and preprocess the dataset
    '''

    # load in the data
    train_data = pd.read_csv('data/training.csv')

    print('loading the dataset')
    features = ['V1','V2','V3','V4','V5','V6','V7']

    #only retain features in the features list
    train_data=preprocess_data(train_data)
    train_data_promote=train_data[train_data['if_promote']!=-1]

    X=train_data_promote[features]
    y=train_data_promote['if_promote']

    return X,y

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),#this is the best
    SVC(gamma=2, C=1), #this is the second best
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features='auto'),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

def downsample_majority(X_train,y_train,n):
    '''downsample the dataset
    '''
    # We downsample our dominant class in our training set
    i_class0 = np.where(y_train == 0)[0]
    i_class1 = np.where(y_train == 1)[0]

    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

    # For every observation of class 0, randomly sample from class 1 without replacement
    i_class0_downsampled = np.random.choice(i_class0, size=n_class1*n, replace=False)

    # Create new indices based on the downsampled class 0
    indices = np.concatenate((i_class0_downsampled, i_class1))
    np.random.shuffle(indices) # we shuffle to avoid order in our dataset

    # We subset our X_train and y_train datasets, we now have a ratio of 2:1 for class 0 vs 1
    X_train = X_train.iloc[indices]
    y_train = y_train.iloc[indices]

    return X_train,y_train

def compare_classifiers(classifiers,X,y):
    '''compare the performance of different classifier
    
    args:
        classifiers(list): a list of sklearn classifier
        X,y: training set
        scoring: metrics for evaluating different classifiers
        
    returns:
        best_classifer: a classifier with highest metrics
    '''
    #split the dataset
    X_train, X_test, y_train, y_test=train_test_split(X,y,stratify=y)
    X_train,y_train=downsample_majority(X_train,y_train,2)

    #fit and evaluate classifiers
    highest_score=0
    best_classifier=DummyClassifier()
    f1_list=[]
    

    for classifier in classifiers:
        print('start fitting classifier {}'.format(classifier.__class__.__name__))

        pipe=Pipeline([('st',StandardScaler()),('clf',classifier)])
        pipe.fit(X_train,y_train)
        y_pre=pipe.predict(X_test)

       
        f1=f1_score(y_test,y_pre)
        f1_list.append(f1)
        if f1>highest_score:

            best_classifier=pipe
            highest_score=f1
    print(f1_list)
    return best_classifier

        
def determine_best_classifier(classifiers):
    '''select the best classifiers

    args:
        classifiers(list): a list of machine learning classifiers        
    '''
    #load and preprocess the dataset
    X,y=prepare_dataset()

    print('determine the best classifier')
    best_classifier=compare_classifiers(classifiers,X,y)
    
    logging.info('best classifier is {}'.format(best_classifier.__class__.__name__))



if __name__ == '__main__':
    determine_best_classifier(classifiers)