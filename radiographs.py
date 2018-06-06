"""
Functions to load and preprocess radiographs
"""

import sys
import os
import hashlib
import cv2
import numpy as nmpy
from scipy.ndimage import morphology
from utils import update_progress


#Loading the radiographs and storing them in an array
def load(path="Data/Radiographs/", indices=range(1,15)):
    files = ["{:02d.tif}" % i if i < 15 else "extra/{:02d.tif}" % i for i in indices]
    imageArray = [cv2.imread(path+f) for f in files]
    
    return imageArray
 
    

#Computing median
def med(input_array, input_length):
        sorted_input = nmpy.sort(input_array)
        computed_median = sorted_input[input_length/2]
        return computed_median   



def adaptive_median_filtering(image_array, window_size, adptive_threshold):
    """
    Applying an adaptive median filter
    
    Args:
        image_array: Source image as numpy array
        window_size: filter window size used in adaptive median filtering
        adptive_threshold: Sets the adaptive threshold for filtering
        
    Returns:
        The image after adaptive median filtering
    
    
    .. _Based on:
        https://github.com/sarnold/adaptive-median/blob/master/adaptive_median.py
    """
    
    copy_array = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2GRAY)
    
    # setting the filter window and dimenions of the image to be filtered
    win = 2*window_size + 1
    y_axis, x_axis = image_array.shape
    v_length = win*win
    
    # create 2-D image array and intialize window
    filter_window = np.array(np.zeros((win, win)))
    target_vector = np.array(np.zeros(v_length))
    pixel_count = 0
    
    try:
        # loop over image with specified window W
        for y in range(window_size, y_axis-(window_size+1)):
            update_progress(y/float(y_axis))
            for x in range(window_size, x_axis-(window_size+1)):
                # populate window, sort, find median
                filter_window = copy_array[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
                target_vector = np.reshape(filter_window, ((v_length),))
                # internal sort
                median = med(target_vector, v_length)
                # check for threshold
                if not adptive_threshold > 0:
                    copy_array[y, x] = median
                    pixel_count += 1
                else:
                    scale = np.zeros(v_length)
                    for n in range(v_length):
                        scale[n] = abs(target_vector[n] - median)
                    scale = np.sort(scale)
                    Sk = 1.4826 * (scale[v_length/2])
                    if abs(copy_array[y, x] - median) > (adptive_threshold * Sk):
                        copy_array[y, x] = median
                        pixel_count += 1
        update_progress(1)

    except TypeError:
        print ("Error in adaptive median filter function")
        sys.exit(2)

    print (pixel_count, "pixel(s) filtered out of", x_axis*y_axis)
    return copy_array
    
    
    
def enhance_img(image_array):
    """
    Applies bilateral filter, top&bottom hat transformations and CLAHE.
    
    Args:
        The image to be enhanced.
    Returns:
        Enhanced image.
    """
    
    img = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 175, 175)
    img_bright = morphology.white_tophat(img, size=400)
    img_dark = morphology.black_tophat(img, size=80)
    
    img = cv2.add(img, img_bright)
    img = cv2.subtract(img, img_dark)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = clahe_obj.apply(img)
    return img
    