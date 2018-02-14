# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:08:51 2018

@author: edwin
"""


from ClassifierWrap import *
from util import *
import numpy as np

## Wrapper for music analysis

## defaultDir is the directory that contains all other training and test directories 
defaultDir='C:/Users/edwin/Documents/music/pyAudioAnalysis-master/musicFile/'

def UI():
    
    ## User Choices ! If None is assigned, default value will be used
    ## window=0.05, step=0.02, directory=defaultDir
    ## Must check parameters carefully
    ## Model Choice : 'SVM' , 'SVM_RBF', 'GradientBoosting','RandomForest','ExtraTrees'
    ###############################################################################################
    listOfMusic=['Classical','Rock','Jazz'];
    window=[0.4,0.4,0.4];windowStep=[0.1,0.1,0.1];
    MIAlgorithm='SVM_RBF';testDir=defaultDir+'Test'
    ###############################################################################################
    trainDir=[defaultDir+mem for mem in listOfMusic]
    R=ClassifierTest(MIAlgorithm,listOfMusic,trainDir,testDir,window,windowStep,RBF1=False)    
    """ View The results"""          
    viewResults(R)
 

UI()