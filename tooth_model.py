from procustes_analysis import procrustes
from pca import pca_code, num_eig
import pltr as plotter
import matplotlib.pyplot as MPL
import numpy as np
from landmarks import Landmarks, get_vectors,load_landmarks
from radiograph import Radiograph,load_radiographs
import os
import utils
from pltr import plot_all_landmarks

class Tooth(object):
    """class for each incisor
    """

    def __init__(self, num, landmarks):
        """new incisor
        Args:
            num (int 1-8): int representing the tooth number
        """
        self.preprocess(landmarks)
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
        [evals,evecs,mu] = pca_code(self.aligned)

        Q = []

        n = len(evals)
        for i in range(0,n):
            Q.append(evals[i]* evecs[:,i])
        self.Q = Q
        self.mu = mu
        self.evals = evals
        self.evecs = evecs

    @staticmethod
    def get_pt_difference(landmarks,n):
        d = np.zeros((n,n))

        for i in range(0,n):
            for j in range(0,n):
                p1 = landmarks.get_coordinate_at_index(i)
                p2 = landmarks.get_coordinate_at_index(j)
                d[i,j] = utils.get_distance(p1,p2)
        return d

    def get_weights(self,landmarks1,landmarks2):
        """ returns a transposed vector of get_weights
        """

        n = landmarks1.get_length()
        d1 = self.get_pt_difference(landmarks1,n)

        if landmarks2 is None:
            d2 = np.zeros((n,n))
        else:
            d2 = self.get_pt_difference(landmarks2,n)

        w = np.zeros((n))
        for i in range(0,n):
            for j in range(0,n):
                w[i] += np.var([d1[i,j],d2[i,j]])
        w = 1/w
        print(w)
        return w


    def model_reconstruction(self,landmarks,img):
        """
        """
        dx = self.get_direction(landmarks,img)
        self.get_weights(dx,None)


    def get_direction(self,landmarks,radiograph):
        """ get a direction for an edge for the model reconstruction
        """
        X = []
        for index,landmark in enumerate(landmarks):
            norm = self.normalised(landmarks,index)
            x = landmark.get_edge(norm,radiograph)
            X.append(x)
        dx = Landmarks(X)
        return dx

    @staticmethod
    def normalised(landmarks,index):
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
        tooth = Tooth(num,landmarks)
        tooth.preprocess(landmarks)
        tooth.ASM(tooth.aligned)
        tooth.model_reconstruction(tooth.aligned,processed)
        # tooth.get_direction(tooth.aligned,processed)
