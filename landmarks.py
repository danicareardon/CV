import numpy as np
import os

class Landmarks(object):
    """class for loading and preprocessing landmarks
    TODO: say more abouy class
    """

    def __init__(self, data):
        """class for loading and preprocessing landmarks

        Args:
            data: TODO
        """
        print("todo")

    def load(directory, incisor, mirrored):
        """loads all models for an incisor

        Args:
            directory: directory where the incisors are located
                ("/_Data/Landmarks/")
            incisor: the incisor identifier (number)
            mirrored: boolean if mirrored or not

        Returns: a list of all landmark models
        """
