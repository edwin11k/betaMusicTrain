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
    ###############################################################################################
    listOfMusic=['Classical','Rock'];
    trainDir=[defaultDir+listOfMusic[0],defaultDir+listOfMusic[1]];
    window=[0.2,0.2];windowStep=[0.05,0.05];
    MIAlgorithm='KNN';testDir=defaultDir+'Test'
    ###############################################################################################
    

    """ Binary Linear SVM Classifier"""
    if MIAlgorithm=='SVM':
        print('Binary Linear SVM Classifier')
        model=binaryMusicSVM(listOfMusic,trainDir,window,windowStep,RBF=False)
    
        
    """ Binary RBF SVM Classifier"""
    if MIAlgorithm=='SVM_RBF':
        print('Binary RBF SVM Classifier')
        model=binaryMusicSVM(listOfMusic,trainDir,window,windowStep,RBF=True)
        
    """ KNN Model Classifier """
    
    if MIAlgorithm=='KNN':
        print ('K- Nearest Neighbor Classifier(currently only for binary purpose)')
        model=binaryMusicKNN(listOfMusic,trainDir,window,windowStep,5)
    
    
 
    ### All files in test directory will be tested for classification#################
    """ Testing files for classification########################################################"""        
    R=testMusicSampleSVM(model,testDir,listOfMusic)
    for mem in R:
        print (mem)
 

    
    

        
def testFile():
    ###########################################################################################################
    listOfMusic=['Classical','Rock'];
    trainDir=[defaultDir+listOfMusic[0],defaultDir+listOfMusic[1]];
    window=[1,1];windowStep=[0.04,0.02];
    MIAlgorithm='SVM';testDir=defaultDir+'Test'
    ###########################################################################################################
    
    newFactory=musicFactory();testMusic=newFactory.loadMusic('test',testDir)
    for i,fileName in enumerate(testMusic.fileName):
        print(testMusic.fileName[i])
    
    
    



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
    


def binaryMusicKNN(listMusic,listDir,window,step,K):
    print('KNN Model Training Initiated! This may take a while~')
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
    knn=trainKNN(features,K)
    print('KNN Training Completed!')
    return knn


def binaryMusicSVM(listMusic,listDir,window,step,RBF):
    print('SVM Model Training Initiated! This may take a while~')
    features=featureStack(loadMusicFeatures(listMusic,listDir,window,step))
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
            print('Warning : New Step and/or Window size assigned!')
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
#testFile()