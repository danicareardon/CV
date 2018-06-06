import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

class Radiograph(object):
        """class for each radiograph image
        """

        def __init__(self,path):
            """new radiograph

            Args:
                path (str): the path of the radiograph image
            """
            self.path = path
            self.img = cv2.imread(path,cv2.CV_8UC1)

        def plot_img(self):
            cv2.imshow('radiograph',self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def preprocess(self):
            self.__grayscale()
            self.plot_img()
            reduced_noise = self.__noisereduction()
            self.plot_img()

        def get_path(self):
            """ getter function for the path
            """
            return self.path

        def get_current_img(self):
            """ getter function for the current image
            """
            return self.img

        def __reset_img(self):
            """ reset the self.img to the __get_original_img
            """
            self.img = cv2.imread(self.path,cv2.CV_8UC1)

        def __grayscale(self):
            """ changes self.img to __grayscale
            """
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        def __noisereduction(self):
            """ performs fastNlMeansDenoising on self.img
            """
            return cv2.fastNlMeansDenoising(self.img,None, 4, 7, 35)



def load_radiographs(x,y):
    """ loads all radiographs images for a give range(x,y)

    Args:
        x (int) : x > 0
        y (int) : y < 31
    """
    paths = []
    for i in range(x,y):
        if i < 15 and i > 0:
            dir = join(".", "_Data/Radiographs/")
            name = "%02d.tif" % i
            if isfile(join(dir,name)):
                paths.append(join(dir,name))
        elif i > 14 and i < 31:
            dir = join(".", "_Data/Radiographs/extra/")
            name = "%02d.tif" % i
            if isfile(join(dir,name)):
                paths.append(join(dir,name))
    imgs = [Radiograph(path) for path in paths]
    return imgs

if __name__ == "__main__":
    imgs = load_radiographs(1,2)
    img = imgs[0]
    print(img.get_path())
    img.preprocess()
