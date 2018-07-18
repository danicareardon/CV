"""Incisor Segmentation main function

Author: Danica Reardon
Student Number: r0604915

Author: Akash Madhusudan
Student Number: r0692878

Python 3.5
"""

import os
from landmarks import load_landmarks, Landmarks
from tooth_model import Tooth, resize
from radiograph import Radiograph
from global_var import grey_profile_pixel, grey_profile_search
import cv2
incisor_index = 8
image_indices = list(range(1,15))
test_set_images = list(range(1,2))
all_incisor_images = list(range(1,9))
test_image_index = 5
import pickle
import numpy as np
from evaluate_fit import evaluate_landmark
scores_p_r_f = np.zeros([len(test_set_images), len(all_incisor_images),3])

def load_images(indices=list(range(1, 15))):
    directory1 = os.path.join(".", "C:/Users/Akash/PycharmProjects/CV/CV/_Data/PreprocessedImages/")
    filenames = ["%02d.tif" % i for i in indices]
    filenames = [os.path.join(os.path.dirname(__file__),directory1, f) for f in filenames]
    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in filenames]
    return images  

def main():
    directory = os.path.join(".", "C:/Users/Akash/PycharmProjects/CV/CV/_Data/Landmarks/")
    all_images = load_images(image_indices)
    for test_image_index in test_set_images:
        train_image_indices = image_indices.copy()
        train_image_indices.remove(test_image_index)
        train_images = [all_images[index-1] for index in train_image_indices]
        test_image = all_images[test_image_index-1]
        current_teeth_model = []
        if 1:
            for incisor_index in all_incisor_images:
                all_landmarks = load_landmarks(directory, incisor_index, 0)
                train_landmarks = [all_landmarks[index-1] for index in train_image_indices]
                test_landmark = all_landmarks[test_image_index-1]
                #building model
                teeth_model_temp = Tooth(incisor_index, train_landmarks)
                current_teeth_model.append(teeth_model_temp)
                current_teeth_model[incisor_index-1].derive_model(train_images, train_landmarks, grey_profile_pixel)
        landmark_initial=[]
        landmark_initial = get_initial_landmarks(test_image, current_teeth_model, landmark_initial, 1)
        landmark_initial = get_initial_landmarks(test_image, current_teeth_model, landmark_initial, 0)  
        img = test_image.copy()
        for inscisor_index in all_incisor_images:
            landmark_final = current_teeth_model[inscisor_index-1].fit(landmark_initial[inscisor_index-1], test_image, grey_profile_search)
            #add the landmark to the image
            #img = preprocess(img)
            colors = [(255, 255, 255), (0,0 ,255 )]
            #colors = [(0, 0, 255)]
            indices_temp = list(range(1,15))
            all_landmarks2 = load_landmarks(directory, indices_temp, 0)
            out = "C:/Users/Akash/PycharmProjects/CV/CV/_Data/out/"
            for ind, lms in enumerate([all_landmarks2[test_image_index-1], landmark_final]):
            #for ind, lms in enumerate([landmark_final]):
                points = lms.coordinates
                for i in range(len(points) - 1):
                    cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                             (int(points[i + 1, 0]), int(points[i + 1, 1])),
                             colors[ind],6)
            scores_p_r_f[test_image_index-1,inscisor_index-1,:] = evaluate_landmark(landmark_final, test_image_index, inscisor_index)
        cv2.imwrite('%s/%02d-all_final_2.png' % (out, test_image_index), img)
        print(scores_p_r_f[test_image_index-1,:,:])
        
    
def get_initial_landmarks(img, current_teeth_model, initial_landmarks, is_upper):
    #resixe the image for display and taking ROI
    ht = img.shape[0]
    test_image = img.copy()
    test_image, scale = resize(test_image, 1200, 800)
    ht1 = test_image.shape[0]
    #taking ROI
    fromCenter = False
    cv2.namedWindow('image_win')
    r = cv2.selectROI('image_win',test_image,fromCenter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #convert the obtained points to true scale.
    r2 = [0,0,0,0]
    r2[0] = r[0] / scale
    r2[1] = r[1] / scale
    r2[2] = (r[0] + r[2]) / scale
    r2[3] = (r[1] + r[3]) / scale
    #scale the mean landmark to the test image size 
    #assuming each teeth has same size, get mid points for each teeth
    for ind in range(1,5):
        if is_upper:
            ind2 = ind
        else:
            ind2 = ind + 4
        mean_landmark = Landmarks(current_teeth_model[ind2-1].mean_shape.coordinates)
        min_x = abs(mean_landmark.coordinates[:, 0].min())
        min_y = abs(mean_landmark.coordinates[:, 1].min())
        scaled_points = [((point[0]+min_x)*ht1, (point[1]+min_y)*ht1) for point in mean_landmark.coordinates]
        mean_landmark.set_coordinates(scaled_points)
        roi = [(r2[0] +(ind-1)*(r2[2]-r2[0])/4, r2[1]), (r2[0] +(ind)*(r2[2]-r2[0])/4, r2[3])]
        #get mid point
        mid_point = np.mean(roi, axis=0)
        mid_point = mid_point
        meanTestLandmark = np.mean(scaled_points, axis = 0)
        delta = mid_point - meanTestLandmark
        #translate the mean_landmark to the mid point
        temp_obj = Landmarks()
        temp_obj = mean_landmark.translate(delta)
        #temp_obj.set_points(temp_obj.points * ht/ht1)
        initial_landmarks.append(temp_obj)
#        temp_points = initial_landmarks[0].points
#        temp_img = img.copy()
#        for j in range(0,int(initial_landmarks[0].points.size/2)):
#            cv2.line(temp_img,(int(temp_points[j,0]), int(temp_points[j,1])),(int(temp_points[(j+1)%40,0]), int(temp_points[(j+1)%40,1])),(255,0,0),2)
#        cv2.imshow('image',temp_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    #end of loop
    #return the list of all initial points.
    return initial_landmarks

if __name__ == "__main__":
    main()
