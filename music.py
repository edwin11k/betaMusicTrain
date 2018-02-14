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
            if music_genre=='Speech':
                print(music_genre+" File Read Located At :"+dirPath+os.sep+file)
            else:
                print(music_genre+" Music File Read Located At :"+dirPath+os.sep+file)
            self.fileData.append(x);self.Fs.append(fs)
            self.stFeatures.append(stFeatureExtraction(x,fs,window*fs,step*fs)) 
            self.fileName.append(file)

    def __str__(self):    
        return 'music'
    
