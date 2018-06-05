from procustes_analysis import procrustes
from pca_model import PCA

class Tooth(objects):
    """class for each incisor
    """

    def __init__(self, num):
        """new incisor

        Args:
            num (int 1-8): int representing the tooth number
        """
        self.num = num
