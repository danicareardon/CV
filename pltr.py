import numpy as np
import matplotlib.pyplot as plt
import cv2
from landmarks import Landmarks

def plot_all_landmarks(landmarks, num):
    """ plots the landmarks

    Args:
         landmarks (Landmarks): Landmarks object of incisors
    """

    minX, minY = [],[]
    maxX, maxY = [],[]

    for landmark in landmarks:
        X,Y = landmark.get_min()
        minX.append(X)
        minY.append(Y)

        X,Y = landmark.get_max()
        maxX.append(X)
        maxY.append(Y)

    X = int(max(maxX))-int(min(minX))+10
    Y = int(max(maxY))-int(min(minY))+10

    img = np.zeros((Y,X))

    for landmark in landmarks:
        coordinates = landmark.get_matrix()
        for i in range(0,len(coordinates)):
            img[int(coordinates[i,1] - min(minY)) + 5, int(coordinates[i,0] - min(minX)) + 5] = 1

    cv2.imshow('Landmarks ' + str(num), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
