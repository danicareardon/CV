from procustes_analysis import procrustes
from pca import pca_code, num_eig
import pltr as plotter
import matplotlib.pyplot as MPL
import numpy as np
from landmarks import Landmarks, get_vectors,load_landmarks
from radiograph import Radiograph,load_radiographs
import os
import utils

class Tooth(object):
    """class for each incisor
    """

    def __init__(self, num):
        """new incisor

        Args:
            num (int 1-8): int representing the tooth number
        """
        self.num = num

    def preprocess(self, landmarks):
        """performs procustes analysis on the landmarks

            Args:
                landmarks (list[Landmarks]) : all landmarks for an incisors

            Returns:
                mean_shape ([Landmarks]) : estimate of the mean_shape
                aligned (List[Landmarks]) : all landmarks aligned to the mean
        """
        self.mean_shape, self.aligned = procrustes(landmarks,self.num)

    def ASM(self,landmarks):
        """performs ASM on the models

            Args:
                landmarks (list[Landmarks]) : all landmarks for an incisors

        """
        # preprocess using procrustes
        self.preprocess(landmarks)

        # perform PCA analysis
        evals,evecs,mu = pca_code(self.aligned)
        num = num_eig(0.98,evals)

        Q = []
        for i in range(0,num):
            Q.append(evals[i]*evecs[:,i])
        self.Q = np.array(Q).squeeze()
        self.mu = mu


    def model_reconstruction(self,landmarks):
        """ reconstructs the model based on the Model Reconstruction
            summarization using the least squares method
        """
        landmarks = get_vectors(landmarks)
        y = np.subtract(landmarks,self.mu).squeeze()
        c = np.linalg.lstsq(y.T,self.Q.T,rcond=None)[0].squeeze()
        self.c = c

    def get_direction(self,landmarks,radiograph):
        """ get a direction for an edge for the model reconstruction
        """
        X = []
        for index,landmark in enumerate(landmarks):
            norm = self.normalised(landmarks,index)
            x = landmark.get_edge(norm,radiograph)
            X.append(x)
        return X

    def normalised(self,landmarks,index):
        max = len(landmarks)
        next = (index+1)%max
        prev = (index-1)%max

        if utils.line(landmarks[prev],landmarks[index],landmarks[next],0):
            return np.array([1,0])
        elif utils.line(landmarks[prev],landmarks[index],landmarks[next],1):
            return np.array([0,1])
        else:
            return utils.normalized(landmarks[prev],landmarks[index],landmarks[next])


if __name__ == "__main__":
    directory = os.path.join(".", "_Data/Landmarks/")

    imgs = load_radiographs(1,2)
    for i in imgs:
        i.preprocess()
        processed = i.sobel

    for num in range(1, 2):
        # 1.1 load landmarks
        landmarks = load_landmarks(directory, num, mirrored=False)
        tooth = Tooth(num)
        tooth.preprocess(landmarks)
        # tooth.ASM(tooth.aligned)
        # tooth.model_reconstruction(tooth.aligned)
        # tooth.get_direction(tooth.aligned,processed)
