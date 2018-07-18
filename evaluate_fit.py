import cv2
import os
import numpy as np
from landmarks import Landmarks
def evaluate_landmark(landmark, image_index, teeth_index):

    """
        Evaluate_landmark corresponding to teeth into image against Truth result
        returns (Precision, Recall, F) as accuracy measures
    """
    dir = os.path.join(".", "_Data/Segmentations/")

    print(teeth_index)
    print(image_index)

    path = str("%02d" % image_index) + "-"
    file = path + str(teeth_index - 1) + ".png"

    file = os.path.join(dir,file)
    print(file)

    if os.path.isfile(file):
        print(file)

    truth = cv2.imread(file, cv2.CV_8UC1)

    #truth = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)

    blank_image = np.zeros(truth.shape, np.uint8)

    landmark_p = np.array(landmark.coordinates, dtype=np.int32)

    try:
        evaluation = cv2.fillPoly(blank_image, [landmark_p], 255)
    except:
        print(" error ")

    # Computing result accuracy
    TP = TN = FP = FN = 1

    for x in range(truth.shape[0]):
        for y in range(truth.shape[1]):
            if (truth[x, y] == 0) and (evaluation[x, y] == 0):
                TN = TN + 1
            elif (truth[x, y] != 0) and (evaluation[x, y] != 0):
                TP = TP + 1
            elif (truth[x, y] == 0) and (evaluation[x, y] != 0):
                FP = FP + 1
            else :#(truth[x, y] != 0) and (evaluation[x, y] == 0)
                FN = FN + 1

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F = (2 * Precision * Recall) / (Precision + Recall)
    return (Precision, Recall, F)
