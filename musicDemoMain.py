# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:08:51 2018

@author: edwin
"""

from binaryClassifierWrap import *
from util import *
import numpy as np

## Wrapper for music analysis

## defaultDir is the directory that contains all other training and test directories 
defaultDir='C:/Users/edwin/Documents/music/pyAudioAnalysis-master/musicFile/'

def binaryUI():
    
    ## User Choices ! If None is assigned, default value will be used
    ## window=0.05, step=0.02, directory=defaultDir
    ## Must check parameters carefully
    ## Model Choice : 'SVM' , 'SVM_RBF', 'GradientBoosting','RandomForest','ExtraTrees'
    ###############################################################################################
    listOfMusic=['Classical','Rock'];
    trainDir=[defaultDir+listOfMusic[0],defaultDir+listOfMusic[1]];
    window=[0.2,0.2];windowStep=[0.05,0.05];
    MIAlgorithm='SVM';testDir=defaultDir+'Test'
    ###############################################################################################
    
    R=binaryClassifierTest(MIAlgorithm,listOfMusic,trainDir,testDir,window,windowStep,RBF1=False)    
    """ View The results"""          
    viewResults(R)
        
        
        
        
    
UI()
