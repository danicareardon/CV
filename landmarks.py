import os
import numpy as np
import glob

class Landmarks(object):
    """class for all landmarks of one incisor
    """

    def __init__(self, data):
        """new set of landmarks

        Args:
            data (str || [x_0,y_0,...,x_n,y_n]): The filepath of the landmark file || an array of points
        """
        if data:
            if isinstance(data, str):
                self._read_landmarks(data)
            elif isinstance(data,np.ndarray) and data.shape[1] is 2:
                self.points = data

    def get_centroid(self):
        """Gets the centroid of the points

        Returns:
            [x,y] : the centroid
        """
        return np.mean(self.points, axis=0)

    def translate_to_origin(self):
        """translates model so that the centroid is at the origin

        Returns:
            [x,y] : translated points
        """
        centroid = self.get_centroid()
        points = self.points - centroid
        return Landmarks(points)



    def _read_landmarks(self, file):
        """reads the landmarks from a file

        Args:
            file: path to landmark file
        """
        points = []
        #https://stackoverflow.com/questions/1657299/how-do-i-read-two-lines-from-a-file-at-a-time-using-python
        lines = open(file).readlines()
        for x, y in zip(lines[0::2], lines[1::2]):
            points.append(np.array([float(x),float(y)]))
        self.points = np.array(points)


def load_landmarks(directory, incisor, mirrored):
    """loads all models for one specific incisor

    Args:
        directory: directory where the incisors are located
            ("/_Data/Landmarks/")
        incisor: the incisor identifier (number)
        mirrored: boolean if mirrored or not

    Returns: a Landmarks object for one incisor
    """
    landmarks = []
    files = glob.glob(directory + "original/*-" + str(incisor) + ".txt")
    if mirrored:
        mirrored_files = glob.glob(directory + "mirrored/*-" + str(incisor) + ".txt")
        files += mirrored_files
    for file in files:
        landmarks.append(Landmarks(file))
    return landmarks

if __name__ == "__main__":
    dir = os.path.join(".", "_Data/Landmarks/")

    # tests getting landmarks loaded
    print("TEST1: loading original files")
    files = load_landmarks(dir,1,False)
    # for f in files:
    #     print(str(f))
    print("TEST2: loading original + mirrored files")
    files = load_landmarks(dir,1,True)
    # for f in files:
    #     print(str(f))
