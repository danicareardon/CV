"""
Functions to load and preprocess radiographs
"""

import sys
import os
import hashlib
import cv2
import numpy as nmpy
from scipy.ndimage import morphology


#Loading the radiographs and storing them in an array
def load(path="Data/Radiographs/", indices=range(1,15)):
    files = ["{:02d.tif}" % i if i < 15 else "extra/{:02d.tif}" % i for i in indices]
    imageArray = [cv2.imread(path+f) for f in files]
    
    return imageArray
    

