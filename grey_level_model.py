import numpy as np
import cv2
from scipy import linspace, asarray
import math
from global_var import num_landmarks
class profile(object):
    def __init__(self, image, grad_image, model, point_ind, k):
        self.image = image
        self.grad_image = grad_image
        self.model_point = model.coordinates[point_ind, :]
        self.k = k
        self.normal = self.__calculate_normal(model.coordinates[(point_ind-1) % num_landmarks, :],
                                              model.coordinates[(point_ind+1) % num_landmarks, :])
        self.points, self.samples = self.__sample()


    def __calculate_normal(self, p_prev, p_next):
        n1 = normal(p_prev, self.model_point)
        n2 = normal(self.model_point, p_next)
        n = (n1 + n2) / 2
        return n / np.linalg.norm(n)

    def __sample(self):
        # Take a slice of the image in pos and neg normal direction
        pos_points, pos_values, pos_grads = self.__slice_image2(-self.normal)
        neg_points, neg_values, neg_grads = self.__slice_image2(self.normal)

        # Merge the positive and negative slices in one list
        neg_values = neg_values[::-1]  # reverse
        neg_grads = neg_grads[::-1]  # reverse
        neg_points = neg_points[::-1]  # reverse
        points = np.vstack((neg_points, pos_points[1:, :]))
        values = np.append(neg_values, pos_values[1:])
        grads = np.append(neg_grads, pos_grads[1:])

        # Compute the final sample values
        div = max(sum([math.fabs(v) for v in values]), 1)
        samples = [float(g)/div for g in grads]

        return points, samples
    def __slice_image2(self, direction):
        """Get the coordinates and intensities of ``k`` pixels along a straight
        line, starting in a given point.

        This version doesn't use interpolation, which makes it less accurate, but
        a lot faster too.
        """
        a = asarray(self.model_point)
        b = asarray(self.model_point + direction*self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k+1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k+1))
        values = self.image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        grad_values = self.grad_image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        return coordinates.T, values, grad_values

class grey_level_model(object):

    def __init__(self):
        self.profiles = []
        self.mean_profile = []
        self.covariance = []
        
    def calc_grad(self, img):
        img = cv2.GaussianBlur(img,(3,3),0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    def process(self, images, landmarks, pixel_width):
        #build profile for each landmark point.
        #find the gradient image
        grad_images = [self.calc_grad(img) for img in images]
        for i in range(0, num_landmarks):
            #find the profile ith landmark point
            #loop through all the images
            profiles_temp = []
            for j in range(len(images)):
                temp = (profile(images[j], grad_images[j], landmarks[j], i, pixel_width))
                profiles_temp.append(temp)
            #find average profile for this landmark
            # calculate mean and covariance
            mat = []
            for p in profiles_temp:
                mat.append(p.samples)
            mat = np.array(mat)
            self.profiles.append(np.mean(mat, axis=0))
            self.covariance.append(np.cov(mat, rowvar=0))
    def quality_of_fit(self, samples, landmark_index):
        temp  = (samples - self.profiles[landmark_index]).T \
            .dot(self.covariance[landmark_index]) \
            .dot(samples - self.profiles[landmark_index])
        return (temp)
            
def normal(point1, point2):
    return np.array([point1[1] - point2[1], point2[0] - point1[0]])
