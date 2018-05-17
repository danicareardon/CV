import numpy as np
import glob
import os

class Landmarks(object):
    """place holder for pre-processing landmarks
    """

    def __init__(self, data):
        """preprocessing landmarks

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

    files = glob.glob(directory + "original/*-" + str(incisor) + ".txt")
    if mirrored:
        mirrored_files = glob.glob(directory + "mirrored/*-" + str(incisor) + ".txt")
        files += mirrored_files
    return files

if __name__ == "__main__":
    dir = os.path.join(".", "_Data/Landmarks/")

    # tests getting landmarks loaded
    print("TEST1: loading original files")
    files = load(dir,1,False)
    for f in files:
        print(str(f))
    print("TEST2: loading original + mirrored files")
    files = load(dir,1,True)
    for f in files:
        print(str(f))
