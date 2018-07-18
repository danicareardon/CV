import os
import numpy as np
import glob

class Landmarks(object):
    """class for all landmarks of one incisor
    """

    def __init__(self, data=None):
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
            elif isinstance(data, list):
                coordinates = []
                for (x,y) in data:
                    coordinates.append(np.array([float(x),float(y)]))
                    self.coordinates = np.array(coordinates)

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

    def get_matrix(self):
        """returns the coordinates as a matrix
        """
        return self.coordinates

    def get_min(self):
        """ returns the minimum X and Y values
        """
        X = self.coordinates[:,0].min()
        Y = self.coordinates[:,1].min()
        return X,Y

    def get_max(self):
        """ returns the maximum X and Y values
        """
        X = self.coordinates[:,0].max()
        Y = self.coordinates[:,1].max()
        return X,Y

    def get_coordinate_at_index(self,index):
        """ returns a coordinate for the index
        """
        max = self.get_length()
        if index < max:
            return self.coordinates[index]
        else:
            return self.coordinates[index%max]

    def get_x(self):
        """ returns all X coordinates
        """
        return self.coordinates[:,0]

    def invT(self, t, s, theta):
        return self.translate(-t).scale(1 / s).rotate(-theta)

    def get_y(self):
        """ returns all X coordinates
        """
        return self.coordinates[:,1]

    def get_length(self):
        """ returns the length
        """
        [n,d] = self.coordinates.shape
        return n

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
        
    def set_coordinates(self,coord):
        self.coordinates = coord

    def get_edge(self,normalised,radiograph):
        landmark = (self.get_x()[0],self.get_y()[0])
        centre = landmark
        found = False
        count = 0

        while not found:
            black = False
            for y in range(-2,2):
                for x in range(-2,2):
                    b = radiograph[int(centre[1])+y][int(centre[0] + x)]
                    if b < 0.0:
                        break
                if b < 0.0:
                    break
            found = not (b < 0.0)

            if found:
                break
            else:
                count = (count * (-1)) + 1 if count <= 0 else count * (-1)
                centre = np.rint(landmark + count * normalised)
                centre = centre.astype(int)
        return centre
        
    def translate(self, vec):
        coord = self.coordinates + vec
        temp = Landmarks(coord)
        return temp
    
    def get_crown(self, is_upper):
        if is_upper:
            temp = Landmarks(self.coordinates[10:30, :])
            return temp
        else:
            pnts = np.vstack((self.coordinates[0:10, :], self.coordinates[30:40, :]))
            temp = Landmarks(pnts)
            return temp

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

def get_vectors(landmarks):
    """gets the landmarks to be vectors

    Args:
        landmarks (list[Landmarks]) : list of landmarks
    """
    arr = []
    for landmark in landmarks:
        arr.append(landmark.get_vector())
    return np.array(arr)

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
