import cv2
import numpy as np
import radiograph as rg
from landmarks import Landmarks, load_landmarks
from tooth_model import Tooth
import os
incisor = []
tmp_incisor = []
dragging = False
start_point = (0, 0)

def manual_initialize(test_image, model_landmark):
    global incisor
    points1 = model_landmark.get_matrix()
    test_image_c1 = test_image.copy()
    img_array = np.array(test_image_c1)
    # scale the model_landmark_points to fit the image scale
    height = test_image_c1.shape[0]
    scale = min(float(1000) / test_image_c1.shape[1], float(650) / height)
    min_x = abs(points1[:, 0].min())
    min_y = abs(points1[:, 1].min())
    incisor = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points1]
    points = np.array([(int(p[0]*height), int(p[1]*height)) for p in incisor])
    cv2.polylines(test_image_c1, [points], True, (255, 0, 0))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image',test_image_c1)
    cv2.setMouseCallback('image',__mouse_click, img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    meanTestLandmark = np.mean(incisor, axis = 0)

    print(meanTestLandmark, Landmarks(np.array([[point[0]*height, point[1]*height] for point in incisor])))



def __mouse_click(event, x, y, flags, img):
    global incisor
    global dragging
    global start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and incisor != []:
            __drag(x, y, img)
    elif event == cv2.EVENT_LBUTTONUP:
        incisor = tmp_incisor
        dragging = False

def __drag(x, y, img):
    global tmp_incisor
    height = img.shape[0]
    tmp = np.array(img)
    dx = (x - start_point[0]) / float(height)
    dy = (y - start_point[1]) / float(height)

    points = [(p[0] + dx, p[1] + dy) for p in incisor]
    tmp_incisor = points

    pimg = np.array([(int(p[0] * height), int(p[1] * height)) for p in points])
    cv2.polylines(tmp, [pimg], True, (255, 0, 0))
    cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('choose', 600, 600)
    cv2.imshow('choose', tmp)
    
    



if __name__ == "__main__":
    directory = os.path.join(".", "_Data/Landmarks/")
    imgs = rg.load_radiographs(1,4)
    landmarks = load_landmarks(directory, 2, mirrored=False)
    for i in imgs:
        i.preprocess()
        x = i.sobel
        y = Tooth(2,landmarks)
        X = manual_initialize(x, y.mean_shape)
        print(X)