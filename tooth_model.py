from procustes_analysis import procrustes
from pca import pca_code
import pltr as plotter
import matplotlib.pyplot as MPL

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
        self.reduced, self.evals, self.evecs, self.mu = pca_code(self.aligned)
