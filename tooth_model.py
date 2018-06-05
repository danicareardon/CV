from procustes_analysis import procrustes
from pca_model import PCA
from landmarks import Landmarks

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
        return procrustes(landmarks,self.num)

    def ASM(self,landmarks):
        """performs ASM on the models

            Args:
                landmarks (list[Landmarks]) : all landmarks for an incisors

        """
        mean_shape, aligned = self.preprocess(landmarks)
