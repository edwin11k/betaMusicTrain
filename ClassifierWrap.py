# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:09:10 2018

@author: edwin
"""

from musicHandler import *
from music import *
from pyAudioAnalysis.audioTrainTest import *
from pyAudioAnalysis.audioFeatureExtraction import *
import numpy as np
import os
from util import *



def ClassifierTest(MIAlgorithm,listOfMusic1,trainDir1,testDir1,window1,windowStep1,RBF1=None):
        ### Linear SVM Classifier ###
    if MIAlgorithm=='SVM':
        print('Linear SVM Classifier')
        model=multiSVM(listOfMusic1,trainDir1,window1,windowStep1,RBF=False)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
           
    ### RBF SVM Classifier ###
    if MIAlgorithm=='SVM_RBF':
        print('RBF SVM Classifier')
        model=multiSVM(listOfMusic1,trainDir1,window1,windowStep1,RBF=True)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
        
    ### Gradient Boosting ### 
    if MIAlgorithm=='GradientBoosting':
        print ('Gradient Boosting Classifier')
        model=gradientBoosting(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
    
    ### Random Forest Training ###
    if MIAlgorithm=='RandomForest':
        print('Random Forest Classifier')
        model=randomForest(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
        
    ### Extra Trees Training ###
    if MIAlgorithm=='ExtraTrees':
        print('Extra Trees Classifier')
        model=extraTrees(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)        
        
    return R
    

def randomForest(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Random Forest Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    rf=trainRandomForest(features, 100)
    print('Gradient Boosting Training Completed!')
    return rf


def extraTrees(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Extra Trees Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    et=trainExtraTrees(features, 100)
    print('Extra Trees Training Completed!')
    return et

def gradientBoosting(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Gradient Boosting Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    gb=trainGradientBoosting(features, 100)
    print('Gradient Boosting Training Completed!')
    return gb


def multiSVM(listMusic,listDir,window,step,RBF):
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('SVM Model Fitting Initiated! This may take a while~')
    if RBF:
        svm=trainSVM_RBF(features,0.1)
    else:
        svm=trainSVM(features,0.1)
    print('SVM Training Completed!')
    return svm
   
def testMusicSampleSVM(model,testDir,trainTypes):
    tFactory=musicFactory();testMusic=tFactory.loadMusic('testMusic',testDir)
    features=testMusic.stFeatures;resultStr=[]
    for i,feature in enumerate(features):
        predict=[0]*len(trainTypes)
        for j in range(len(feature[1])):
            predict[int(model.predict(feature[:,j].reshape(1,-1)))]+=1
        resultStr.append((testMusic.fileName[i],predict,trainTypes[argmax(predict)]))
    return resultStr     

