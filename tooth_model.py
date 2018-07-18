from procustes_analysis import procrustes
from pca import pca_code, num_eig
import pltr as plotter
import matplotlib.pyplot as MPL
import numpy as np
from landmarks import Landmarks, get_vectors,load_landmarks
from radiograph import Radiograph,load_radiographs
import os
import utils
from pltr import plot_all_landmarks
from grey_level_model import profile
from grey_level_model import grey_level_model
import math
import cv2
from procustes_analysis import align_two_shapes, get_s_and_theta
MAX_ITER = 50

class Tooth(object):
    """class for each incisor
    """

    def __init__(self, num, landmarks):
        """new incisor
        Args:
            num (int 1-8): int representing the tooth number
        """
        self.num = num
        self.preprocess(landmarks)

    def preprocess(self, landmarks):
        """performs procustes analysis on the landmarks
            Args:
                landmarks (list[Landmarks]) : all landmarks for an incisors
            Returns:
                mean_shape ([Landmarks]) : estimate of the mean_shape
                aligned (List[Landmarks]) : all landmarks aligned to the mean
        """
        self.mean_shape, self.aligned = procrustes(landmarks,self.num)

    def ASM(self):
        """performs ASM on the models
            Args:
                landmarks (list[Landmarks]) : all landmarks for an incisors
        """
        # perform PCA analysis
        [evals,evecs,mu] = pca_code(self.aligned)

        Q = []
        temp = []

        n = len(evals)
        for i in range(0,n):
            Q.append(evals[i]* evecs[:,i])
            temp.append(np.sqrt(evals[i]) * evecs[:, i])
        self.Q = Q
        self.pca_modes = np.array(temp).squeeze().T
        self.mu = mu
        self.evals = evals
        self.evecs = evecs

    @staticmethod
    def get_pt_difference(landmarks,n):
        d = np.zeros((n,n))

        for i in range(0,n):
            for j in range(0,n):
                p1 = landmarks.get_coordinate_at_index(i)
                p2 = landmarks.get_coordinate_at_index(j)
                d[i,j] = utils.get_distance(p1,p2)
        return d

    def get_weights(self,landmarks1,landmarks2):
        """ returns a transposed vector of get_weights
        """

        n = landmarks1.get_length()
        d1 = self.get_pt_difference(landmarks1,n)

        if landmarks2 is None:
            d2 = np.zeros((n,n))
        else:
            d2 = self.get_pt_difference(landmarks2,n)

        w = np.zeros((n))
        for i in range(0,n):
            for j in range(0,n):
                w[i] += np.var([d1[i,j],d2[i,j]])
        w = 1/w
        return w

    @staticmethod
    def get_sum(arr):
        return np.sum(arr)

    @staticmethod
    def get_pose_parameters(arr,w):
        """ Args: arr (either all x or all y coordinates)
        weighted array
        """
        c = arr + w


    def model_reconstruction(self,img):
        """
        """
        dx = self.get_direction(self.aligned,img)
        w = self.get_weights(dx,None)
        self.get_pose_parameters(dx.get_x(),w)


    def get_direction(self,landmarks,radiograph):
        """ get a direction for an edge for the model reconstruction
        """
        X = []
        for index,landmark in enumerate(landmarks):
            norm = self.normalised(landmarks,index)
            x = landmark.get_edge(norm,radiograph)
            X.append(x)
        dx = Landmarks(X)
        return dx

    @staticmethod
    def normalised(landmarks,index):
        max = len(landmarks)
        next = (index+1)%max
        prev = (index-1)%max

        if utils.line(landmarks[prev],landmarks[index],landmarks[next],0):
            return np.array([1,0])
        elif utils.line(landmarks[prev],landmarks[index],landmarks[next],1):
            return np.array([0,1])
        else:
            return utils.normalized(landmarks[prev],landmarks[index],landmarks[next])

    def derive_model(self,images,landmarks,num_pixels):
        self.p = num_pixels
        self.preprocess(landmarks)
        self.ASM()
        self.glm = grey_level_model()
        self.glm.process(images,landmarks,num_pixels)

    def calc_grad(self, img):
        img = cv2.GaussianBlur(img,(3,3),0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


    def fit(self, X, test_image, num_pix):
        X = self.__fit_incisor(X, test_image, self.glm, num_pix, MAX_ITER)
        print("Done - fitting teeth model for %d teeth onto test image"%self.num)
        return X

    def __find_fit(self, X, img, grad_img, num_pix):
        fit_arr = []
        profiles = []
        best_s = []
        qual = []
        for i in range(len(X.coordinates)):
            prf = profile(img,grad_img,X,i,num_pix)
            profiles.append(prf)

            #testing the fit
            dmin, best = np.inf, None
            dists = []
            for j in range(self.p, self.p+2*(num_pix-self.p)+1):
                subprof = prf.samples[j-self.p:j+self.p+1]
                dist = self.glm.quality_of_fit(subprof, i)
                dists.append(dist)
                if dist < dmin:
                    dmin = dist
                    best = j

            best_s.append(best)
            qual.append(dmin)
            best_point = [int(c) for c in prf.points[best, :]]

        best_s.extend(best_s)
        for best, prf in zip(best_s, profiles):
            best_point = [int(c) for c in prf.points[best, :]]
            fit_arr.append(best_point)

        is_upper = True if self.num <5 else False
        if is_upper:
            quality = np.mean(qual[10:30])
        else:
            quality = np.mean(qual[0:10] + qual[30:40])
        temp = Landmarks(np.array(fit_arr))
        return temp, quality

    def __fit_incisor(self, X, test_image, glms, num_pix, max_iter):
        gradimg = self.calc_grad(test_image)
        b = np.zeros(self.pca_modes.shape[1])
        X_prev = Landmarks(np.zeros_like(X.coordinates))
        # 4. Repeat until convergence.
        nb_iter = 0
        best = np.inf
        best_Y = None
        total_s = 1
        total_theta = 0
        while (nb_iter <= max_iter):
            # 1. Examine a region of the image around each point Xi to find the
            # best nearby match for the point
            Y, quality = self.__find_fit(X, test_image, gradimg, num_pix)
            if quality < best:
                best = quality
                best_Y = Y
            # no good fit found => go back to best one
            if nb_iter == max_iter:
                Y = best_Y
            # 2. Update the parameters (Xt, Yt, s, theta, b) to best fit the
            # new found points X
            b, t, s, theta = self.__update_fit_params(X, Y, test_image)
            #todo do the transformation of X using b,,t,s, theta
            # 3. Apply constraints to the parameters, b, to ensure plausible
            # shapes
            # We clip each element b_i of b to b_max*sqrt(l_i) where l_i is the
            # corresponding eigenvalue.
            limit1 = -3 * np.sqrt(self.evals)
            limit2 = -1 * limit1
            #b = np.clip(b, -3, 3)
            b = np.clip(b, limit1, limit2)
            # t = np.clip(t, -5, 5)
            # limit scaling
            s = np.clip(s, 0.7, 1.05)
            if total_s * s > 1.20 or total_s * s < 0.8:
                s = 1
            total_s *= s
            # limit rotation
            theta = np.clip(theta, -math.pi/4, math.pi/4)
            if total_theta + theta > math.pi/2 or total_theta + theta < - math.pi/2:
                theta = 0
            total_theta += theta

            # The positions of the model points in the image, X, are then given
            # by X = TXt,Yt,s,theta(X + Pb)
            X_prev = X
            temp_points = (X.get_vector() + np.dot(self.pca_modes, b))
            temp_landmark = self.pose([t[0],t[1], s, theta],temp_points)
            X=temp_landmark
            nb_iter += 1
        return X

    def pose(self, pose_params, pnts=[]):
        if len(pnts):
            temp = np.array((pnts[:40], pnts[40:])).T
        else:
            temp = self.mu.reshape(int(len(self.mu)/2), 2)
        lms = Landmarks(temp)
        [tx,ty,s,a] = pose_params
        lms = lms.rotate(a)
        lms = lms.scale(s)
        lms = lms.translate([tx,ty])
        return lms

    def __update_fit_params(self, X, Y, test_image):
        # 1. Initialise the shape parameters, b, to zero (the mean shape).
        b = np.zeros(self.pca_modes.shape[1])
        b_prev = np.ones(self.pca_modes.shape[1])
        i = 0
        while (np.mean(np.abs(b-b_prev)) >= 1e-14):
            i += 1
            # 2. Generate the model point positions using x = X + Pb
            x = Landmarks()
            temp_vector = X.get_vector() + np.dot(self.pca_modes, b)
            x.coordinates = np.array([temp_vector[:int(len(temp_vector)/2)], temp_vector[int(len(temp_vector)/2):]]).T

            # 3. Find the pose parameters (Xt, Yt, s, theta) which best align the
            # model points x to the current found points Y
            is_upper = True if self.num < 5 else False
            x_temp = x.get_crown(is_upper)
            y_temp = Y.get_crown(is_upper)
            t, s, theta = get_s_and_theta(x_temp,y_temp)
            # 4. Project Y into the model co-ordinate frame by inverting the
            # transformation T
            y = Y.invT(t, s, theta)

            # 5. Project y into the tangent plane to X by scaling:
            # y' = y / (y*X).
            yacc = Landmarks()
            temp_vector2 = y.get_vector() / np.dot(y.get_vector(), X.get_vector().T)
            yacc.coordinates = np.array([ temp_vector2[:int(len(temp_vector2)/2)], temp_vector2[int(len(temp_vector2)/2):]]).T

            # 6. Update the model parameters to match to y': b = PT(y' - X)
            b_prev = b
            b = np.dot(self.pca_modes.T, (yacc.get_vector()-X.get_vector()))

            # 7. If not converged, return to step 2

        return b, t, s, theta

def resize(image, width, height):
    #find minimum scale to fit image on screen
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale

if __name__ == "__main__":
    directory = os.path.join(".", "_Data/Landmarks/")

    imgs = load_radiographs(2,3)
    for i in imgs:
        i.preprocess()
        processed = i.sobel

    for num in range(1, 2):
        # 1.1 load landmarks
        landmarks = load_landmarks(directory, num, mirrored=False)
        tooth = Tooth(num,landmarks)
        tooth.ASM()
        tooth.model_reconstruction(processed)
        # tooth.get_direction(tooth.aligned,processed)
