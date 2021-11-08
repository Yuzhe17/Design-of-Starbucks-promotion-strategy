# load in packages
import numpy as np
import pandas as pd
import scipy as sp
import logging
import datetime
import sys
import glob

from select_classifiers import downsample_majority
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle


logging.basicConfig(filename='starbucks.log', level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
logging.info('start running: {}'.format(str(datetime.datetime.now())))

# load in the data
train_data = pd.read_csv('data/training.csv')

def save_model(model,classifier_path):
    '''save the model as pickle file

    args:
        model: sklearn predictor

    returns:
        classifier_path: path where a trained classifier is stored
    '''
    with open(classifier_path,"wb") as f:
        pickle.dump(model,f)

def optimize_classifier(X,y):
    '''tuning the hyperparameters of linear SVC

    args:
        X, y: ndarray. Training dataset

    returns:
        classifier: a sklearn classifier with fine tuned hyperparameters
    '''

    dt=DecisionTreeClassifier()
    gs=GridSearchCV(dt,param_grid={'max_depth':[3,4,5],'min_samples_split':[2,4,6],'min_samples_leaf':[1,3,5],'max_leaf_nodes':[5,10,15]},scoring='f1')
    
    gs.fit(X,y)

    return gs.best_estimator_
    

def calculate_f1(y_score,y,t):
    '''evaluate the classifier with f1_score using different threshold

    args:
        y_score(float): the probabilities of predictions
        y(float): the target of test set
        t: threshold for determining the target label

    returns:
        f1(float): f1 score
    '''

    y_pre=np.zeros((len(y_score),1))
    y_pre[y_score>t]=1
    f1=f1_score(y,y_pre)

    return f1

def plot_f1_threshold(y_score,y,threshold_list):
    '''plot the f1 score versus threshold and calculate the threshold corresponding 

    args:
        y_score(float): the probabilities of predictions
        y(float): the target of test set
        threshold_list(list): a list of decision threshold as x values of the plot

    returns:
        float: a decision threshold which generates the highest f1 score
    '''

    f1_list=[calculate_f1(y_score,y,t) for t in threshold_list]

    plt.figure()
    plt.plot(threshold_list,f1_list,'g-')
    plt.ylabel("f1 score")
    plt.xlabel("Decision Threshold")
    plt.show()

    return threshold_list[np.argmax(f1_list)]

def calculate_cutoffs(y_score,n):
    '''divide the y_score to several parts and calculate the cutoffs

    args:
        y_score(array): an array of scores to divide
        n(int): number of cutoffs 

    returns:
        cutoffs_list(list): a list of cutoffs
    '''

    counts=len(y_score)//n
    indices = [(i+1)*counts for i in range(n)]
    sorted_score= sorted(y_score)
    cutoffs_list=map(lambda x:sorted_score[x],indices)

    return list(cutoffs_list)

if __name__ == '__main__':
    
    print('loading the dataset')
    features = ['V1','V2','V3','V4','V5','V6','V7']

    #only retain features in the features list
    X=train_data[features]
    y=train_data['purchase']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train,y_train=downsample_majority(X_train,y_train,2)


    if not glob.glob('*.pkl'):
        classifier_path = input('please enter a path for classifier:')
        #save_best_classifier(classifiers,classifier_path) 

        print("optimize the classifier")
        #optimize the DecisionTreeClassifier and the DecisionTree is selected over other classification algorithms after comparison
        classifier = optimize_classifier(X_train,y_train)
        classifier.fit(X_train,y_train)

        print('save the classifier')
        save_model(classifier,classifier_path)
    else:
        print('classfier is already saved') 
        classifier=pickle.load(open(glob.glob('*.pkl')[0],'rb'))

    print('predict the probabilites of possible outcome')
    y_score=classifier.predict_proba(X_test)[:,1]

    print('plot the f1 versus threshold curve')
    threshold=plot_f1_threshold(y_score,y_test,calculate_cutoffs(y_score,20))
    logging.info('optimal decision threshold for max f1: {}'.format(threshold))

    



    