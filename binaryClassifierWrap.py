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


def binaryClassifierTest(MIAlgorithm,listOfMusic1,trainDir1,testDir1,window1,windowStep1,RBF1=None):
        ### Binary Linear SVM Classifier ###
    if MIAlgorithm=='SVM':
        print('Binary Linear SVM Classifier')
        model=binaryMusicSVM(listOfMusic1,trainDir1,window1,windowStep1,RBF=False)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
           
    ### Binary RBF SVM Classifier ###
    if MIAlgorithm=='SVM_RBF':
        print('Binary RBF SVM Classifier')
        model=binaryMusicSVM(listOfMusic1,trainDir1,window1,windowStep1,RBF=True)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
        
    ### Binary Gradient Boosting ### 
    if MIAlgorithm=='GradientBoosting':
        print ('Binary Gradient Boosting Classifier')
        model=binaryGradientBoosting(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
    
    ### Random Forest Training ###
    if MIAlgorithm=='RandomForest':
        print('Random Forest Classifier')
        model=binaryRandomForest(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)
        

    if MIAlgorithm=='ExtraTrees':
        print('Extra Trees Classifier')
        model=binaryExtraTrees(listOfMusic1,trainDir1,window1,windowStep1)
        R=testMusicSampleSVM(model,testDir1,listOfMusic1)        
        
    return R
    

def binaryRandomForest(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Random Forest Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    rf=trainRandomForest(features, 100)
    print('Gradient Boosting Training Completed!')
    return rf


def binaryExtraTrees(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Extra Trees Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    et=trainRandomForest(features, 100)
    print('Extra Trees Training Completed!')
    return et

def binaryGradientBoosting(listMusic,listDir,window,step):  
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    print('Gradient Boosting Model Fitting Initiated! This may take a while~')
    # number of estimator set as 100 as default
    gb=trainGradientBoosting(features, 100)
    print('Gradient Boosting Training Completed!')
    return gb


def binaryMusicSVM(listMusic,listDir,window,step,RBF):
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
        pos=0;neg=0;total=0;
        for j in range(len(feature[1])):
            if int(model.predict(feature[:,j].reshape(1,-1)))==0:
                pos+=1
            else:
                neg+=1
            total+=1
        if pos/total>0.5:
            resultStr.append((testMusic.fileName[i],pos,neg,pos/total,trainTypes[0]))
        else:
            resultStr.append((testMusic.fileName[i],pos,neg,pos/total,trainTypes[1]))

    return resultStr     