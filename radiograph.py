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
            self.img = cv2.imread(path,cv2.CV_8UC1)

        def plot_img(self):
            cv2.imshow('radiograph',self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def plot_test(self,img,name):
            cv2.namedWindow(name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name,600,600)
            cv2.imshow(name,img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def preprocess(self):
            # images = []
            # self.__grayscale()
            # reduced_noise = self.__noisereduction()
            # images.append(reduced_noise)

            # self.plot_img()
            #laplacian = self.__laplacian_filter(reduced_noise)
            #self.plot_test(laplacian, "laplacian")
            # images.append(self.__sobel(reduced_noise))

            # sobel = self.__sobel(reduced_noise)
            # laplacian = self.__laplacian_filter(sobel)
            # self.__testing(images)
            # self.plot_test(sobel,"sobel")
            # self.plot_test(laplacian,"laplacian")
            # laplacian = self.__laplacian_filter(reduced_noise)
            # sobel = self.__sobel(laplacian)
            # self.plot_test(sobel,"sobel")
            # self.plot_test(laplacian,"laplacian")

            img = self.img
            img = self.__noisereduction(img)
            laplacian = self.__laplacian_filter(img)
            self.plot_test(laplacian, "laplacian")
            sobel = self.__sobel(laplacian)
            self.plot_test(sobel, "sobel")
            reduced_noise = self.__noisereduction2(sobel)
            self.plot_test(reduced_noise, "after")




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

        def __laplacian_filter(self,img):
            """ returns an image with a laplacian filter
            """
            return cv2.Laplacian(img,cv2.CV_32F)

        def __sobel(self,img):
            """ returns the sobel detected image based on
            https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
            """
            sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
            sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
            sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
            ksize = np.ones((10,10),np.uint8)
            img_white = cv2.morphologyEx(sobel, cv2.MORPH_TOPHAT, ksize)
            ksize = np.ones((80,80), np.uint8)
            img_black = cv2.morphologyEx(sobel, cv2.MORPH_BLACKHAT, ksize)

            image = cv2.add(sobel,img_white)
            image = cv2.subtract(sobel, img_black)

            return image

                # img_bright = morphology.white_tophat(img, size=400)
                # img_dark = morphology.black_tophat(img, size=80)
                #
                # img = cv2.add(img, img_bright)
                # img = cv2.subtract(img, img_dark)
                # clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
                # img = clahe_obj.apply(img)

        def __noisereduction2(self,img):
            """ performs fastNlMeansDenoising on self.img
            """
            # decided to test 9 x 9 with the default BiLateral
            # gaussian = cv2.GaussianBlur(img,(3,3),0)
            # median = cv2.medianBlur(img,9)
            return cv2.GaussianBlur(img,(9,9),0)

        def __noisereduction(self,img):
            """ performs fastNlMeansDenoising on self.img
            """
            # decided to test 9 x 9 with the default BiLateral
            # gaussian = cv2.GaussianBlur(img,(3,3),0)
            median = cv2.medianBlur(img,9)
            return cv2.bilateralFilter(median,9,75,75)

            # N1Means = cv2.fastNlMeansDenoising(self.img,None, 4, 7, 35)
            # self.plot_test(N1Means, "n1")
            # image = []
            # image.append(self.img)
            # Guassian = cv2.GaussianBlur(self.img,(5,5),0)
            # BiLateral = cv2.bilateralFilter(Guassian,9,75,75)
            # image.append(Guassian)
            # image.append(BiLateral)
            # self.__testing(image)

            # image = []
            # # BiLateral = cv2.bilateralFilter(Guassian,9,75,75)
            # image.append(self.img)
            # image.append(cv2.GaussianBlur(self.img,(5,5),0))
            # image.append(cv2.GaussianBlur(self.img,(9,9),0))
            # image.append(cv2.GaussianBlur(self.img,(13,13),0))
            # self.__testing(image)
            # self.plot_test(Guassian, "Guassian")
            # # Median = cv2.medianBlur(self.img,5)
            # # self.plot_test(Median, "Median")
            # BiLateral =
            # self.plot_test(BiLateral, "BiLateral")
            #
            # return cv2.fastNlMeansDenoising(self.img,None, 4, 7, 35)

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
    img = imgs[0]
    print(img.get_path())
    img.preprocess()
