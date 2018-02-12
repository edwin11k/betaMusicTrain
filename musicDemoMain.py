# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:08:51 2018

@author: edwin
"""

from musicHandler import *
from music import *
from pyAudioAnalysis.audioTrainTest import *
from pyAudioAnalysis.audioFeatureExtraction import *
import numpy as np
import os

## Wrapper for music analysis

#defaultDir='C:/Users/edwin/Documents/music/pyAudioAnalysis-master/musicFile/binarySpeech/'
defaultDir='C:/Users/edwin/Documents/music/pyAudioAnalysis-master/musicFile/'

def UI():
    
    ## User Choices ! If None is assigned, default value will be used
    ## window=0.05, step=0.02, directory=defaultDir (in musicHandler.py)
    ## Must check parameters carefully
    ## Model Choice : 'SVM' , 'SVM_RBF', 'GradientBoosting','RandomForest','ExtraTrees'
    ###############################################################################################
    listOfMusic=['Classical','Rock'];
    trainDir=[defaultDir+listOfMusic[0],defaultDir+listOfMusic[1]];
    window=[0.2,0.2];windowStep=[0.05,0.05];
    MIAlgorithm='ExtraTrees';testDir=defaultDir+'Test'
    ###############################################################################################
    

    ### Binary Linear SVM Classifier ###
    if MIAlgorithm=='SVM':
        print('Binary Linear SVM Classifier')
        model=binaryMusicSVM(listOfMusic,trainDir,window,windowStep,RBF=False)
        R=testMusicSampleSVM(model,testDir,listOfMusic)
           
    ### Binary RBF SVM Classifier ###
    if MIAlgorithm=='SVM_RBF':
        print('Binary RBF SVM Classifier')
        model=binaryMusicSVM(listOfMusic,trainDir,window,windowStep,RBF=True)
        R=testMusicSampleSVM(model,testDir,listOfMusic)
        
    ### Binary Gradient Boosting ### 
    if MIAlgorithm=='GradientBoosting':
        print ('Binary Gradient Boosting Classifier')
        model=binaryGradientBoosting(listOfMusic,trainDir,window,windowStep)
        R=testMusicSampleSVM(model,testDir,listOfMusic)
    
    ### Random Forest Training ###
    if MIAlgorithm=='RandomForest':
        print('Random Forest Classifier')
        model=binaryRandomForest(listOfMusic,trainDir,window,windowStep)
        R=testMusicSampleSVM(model,testDir,listOfMusic)
        

    if MIAlgorithm=='ExtraTrees':
        print('Extra Trees Classifier')
        model=binaryExtraTrees(listOfMusic,trainDir,window,windowStep)
        R=testMusicSampleSVM(model,testDir,listOfMusic)        
        
    ### All files in test directory will be tested for classification#################
    """ View The results"""        
    
    viewResults(R)
 


    

def viewResults(results):
    for mem in results:
        print (mem)



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
        
    

def loadMusicFeatures(listMusic,listDir,window,windowStep):
    musicFeatures=[]
    newFactory=musicFactory()
    for i, mem in enumerate(listMusic):
        if window[i]==None and windowStep[i]==None:
            Music=newFactory.loadMusic(mem,listDir[i])
        else:
            print('Warning : New Step and Window size assigned!')
            Music=newFactory.loadMusic(mem,listDir[i],window[i],windowStep[i])
        musicFeatures.append(Music.stFeatures)   
    print('Music Features are loaded')

    return musicFeatures


# Stack the features from various sources in the genre, & Transpose is needed to use list of features
def featureStack(features):
    matrixFeatures=[]
    for feat in features:
        matrixFeatures.append((np.hstack(feat)).T)
    return matrixFeatures
        
        
        
        
    
UI()
