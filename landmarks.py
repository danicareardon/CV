import os
import numpy as np
import glob

class Landmarks(object):
    """class for all landmarks of one incisor
    """

    def __init__(self, data):
        """new set of landmarks

        Args:
            data (str || [x_0,y_0,...,x_n,y_n]): The filepath of the landmark file || an array of coordinates
        """
        if data is not None:
            if isinstance(data, str):
                self._read_landmarks(data)
            elif isinstance(data, np.ndarray) and np.atleast_2d(data).shape[0] == 1:
                lngth = int(len(data)/2)
                self.coordinates = np.array((data[:lngth], data[lngth:])).T
            elif isinstance(data,np.ndarray) and data.shape[1] == 2:
                self.coordinates = data

    def get_centroid(self):
        """Gets the centroid of the coordinates

        Returns:
            [x,y] : the centroid
        """
        return np.mean(self.coordinates, axis=0)

    def translate_to_origin(self):
        """translates model so that the centroid is at the origin

        Returns:
            [x,y] : translated coordinates
        """
        centroid = self.get_centroid()
        coordinates = self.coordinates - centroid
        return Landmarks(coordinates)

    def scale_to_one(self):
        """scales the landmark so that the normalized shape is 1

        Returns:
            [x,y] : coordinates normalized to 1
        """
        centroid = self.get_centroid()
        factor = np.sqrt(np.power(self.coordinates - centroid, 2).sum())
        coordinates = self.coordinates.dot(1. / factor)
        return Landmarks(coordinates)

    def scale(self,value):
        """scales the landmark to value

        Returns:
            [Landmarks] : scaled landmarks
        """
        centroid = self.get_centroid()
        coordinates = (self.coordinates - centroid).dot(value) + centroid
        return Landmarks(coordinates)

    def rotate(self,theta):
        """rotates the landmark to theta

        Returns:
            [Landmarks] : rotates landmarks
        """
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])

        coordinates = np.zeros_like(self.coordinates)
        centroid = self.get_centroid()
        translated = self.coordinates - centroid
        for i in range(len(translated)):
            coordinates[i,:] = translated[i, :].dot(R)
        coordinates = coordinates + centroid
        return Landmarks(coordinates)



    def get_vector(self):
        """returns the coordinates as a vector

        Returns:
            [x0,y0,x1,y1,...xn,yn]
        """
        return np.hstack((self.coordinates[:,0],self.coordinates[:,1]))


    def _read_landmarks(self, file):
        """reads the landmarks from a file

        Args:
            file: path to landmark file
        """
        coordinates = []
        #https://stackoverflow.com/questions/1657299/how-do-i-read-two-lines-from-a-file-at-a-time-using-python
        lines = open(file).readlines()
        for x, y in zip(lines[0::2], lines[1::2]):
            coordinates.append(np.array([float(x),float(y)]))
        self.coordinates = np.array(coordinates)


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
