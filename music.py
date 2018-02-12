#from pyAudioAnalysis.audioVisualization import *
from pyAudioAnalysis.audioBasicIO import *
import matplotlib.pyplot as plt
from pyAudioAnalysis.audioFeatureExtraction import *
from pyAudioAnalysis.audioSegmentation import *
from pyAudioAnalysis.audioVisualization import *
from pyAudioAnalysis.audioTrainTest import *
import os




class Music(object): 
    """ Music object reading files
    S : Speech file 
    C : Classical Music file
    R : Rock Music File
    
    Within the given directory, the class will read all the files starts with the string & save
    
    Training files must be named to address the starting file
    For example classical file in the directory must start with C 
    """
    
    
    def __init__(self,dirPath=os.curdir,music_genre=None,window=0.05,step=0.02):
        self.genre=music_genre
        self.fileData=[]
        self.Fs=[]
        self.stFeatures=[]
        self.fileName=[]
        for file in os.listdir(dirPath):
            [fs,x]=readAudioFile(dirPath+os.sep+file);x=stereo2mono(x)
            print(music_genre+" Music File Read Located At :"+dirPath+os.sep+file)
            self.fileData.append(x);self.Fs.append(fs)
            self.stFeatures.append(stFeatureExtraction(x,fs,window*fs,step*fs)) 
            self.fileName.append(file)

    def __str__(self):    
        return 'music'
    
  
### this part is commented out, but still left for futi
    
#""" Made inheretence structure only because of future scalability
#    Adding another type of class should be easier
#    Directory may be set differently depending on music type. hence inhereted
#    Window & step may be changed for each music type.. not sure if its good idea though
#    (I am very ignorant of music..please bear with me)
#"""
#class Rock(music):
#        
#    def __init__(self,dirPath=os.curdir,music_genre='R',window=0.05,step=0.02):
#        super(Rock,self).__init__(dirPath,music_genre,window,step)
#        self.genre='Rock'
#             
#    def __str__(self):
#        return 'Rock'  
#    
#
