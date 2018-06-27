import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

# todo: Sobel edge detection due to
# https://www.sciencedirect.com/science/article/pii/S2213020916301094

# sobel + gaussian + laplacian filter

class Radiograph(object):
        """class for each radiograph image
        """

        def __init__(self,path):
            """new radiograph

            Args:
                path (str): the path of the radiograph image
            """
            self.path = path
            self.img = cv2.imread(path,0)

        def plot_img(self,image,name):
            img = image.copy()
            img = cv2.resize(img, (432, 228))
            cv2.imshow('radiograph',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("Results/Radiographs/" + name, img);

        def plot_test(self,img,name):
            cv2.namedWindow(name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name,600,600)
            cv2.imshow(name,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def preprocess(self):
            img = self.img
            img = self.__noisereduction(img)
            img = self.__sobel(img)
            self.sobel = img
            #self.plot_test(img," ")

        def gaussian_pyramid(self,level):
            """ creates a guassian image pyramid on self.img
            """
            G = self.img.copy()
            gp = [G]
            for i in range(0,level+1):
                G = cv2.pyrDown(G)
                gp.append(G)
            self.plot_test(G,"1")
            return G

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

        def __grayscale(self,img):
            """ changes self.img to __grayscale
            """
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        def __laplacian_filter(self,img):
            """ returns an image with a laplacian filter
            """
            return cv2.Laplacian(img,cv2.CV_64F)

        def __sobel(self,img):
            """ returns the sobel detected image based on
            https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
            """
            sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
            sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
            gradx = cv2.convertScaleAbs(sobelx)
            grady = cv2.convertScaleAbs(sobely)
            return cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        def __hat_morphology(self,img):
            """ gets the top and bottom hat morphology
            """
            ksize = np.ones((10,10),np.uint8)
            img_white = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, ksize)
            ksize = np.ones((80,80), np.uint8)
            img_black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ksize)
            image = cv2.add(img,img_white)
            image = cv2.subtract(img, img_black)
            return image

        def __noisereduction(self,img):
            """ performs fastNlMeansDenoising on self.img
            """
            img = cv2.GaussianBlur(img,(5,5),0)
            img = cv2.bilateralFilter(img, 9, 200, 200)
            img = self.__hat_morphology(img)
            return img

        def __clahe(self,img):
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(14, 14))
            return clahe_obj.apply(img)

        def __testing(self,images):
            """ displays a list of images
            """
            image = np.concatenate(images, axis=1)
            image = cv2.resize(image, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
            cv2.imshow("test", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
    for i in imgs:
        i.plot_img(i.img,"unfiltered.jpg")
        i.preprocess()
        i.plot_img(i.sobel,"filtered.jpg")
        x = i.gaussian_pyramid(2)
