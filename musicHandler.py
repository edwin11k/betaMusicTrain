# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:35:01 2018

@author: edwin
"""
""" This file will get music!!!! """

from music import *
import os

class musicFactory(object):
     
       
    def loadMusic(self,music_genre=None,directory=os.curdir,window1=0.05,step1=0.02):
        return Music(dirPath=directory,music_genre=music_genre,window=window1,step=step1)
#        if music_genre=='R':
#            return Rock(dirPath=directory,window=window1,step=step1)
#        elif music_genre=='S': 
#            return Speech(dirPath=directory,window=window1,step=step1)     
#        elif music_genre=='C':  
#            return Classical(dirPath=directory,window=window1,step=step1)
#        elif music_genre=='J':  
#            return Jazz(dirPath=directory,window=window1,step=step1)
#        else:
#            print('Please provide correct music genre you want to load!')
#            print('R: Rock, C:Classical, J:Jazz, S: Human Speech Sample')
#            
    

    








