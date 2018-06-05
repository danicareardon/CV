"""Incisor Segmentation main function

Author: Danica Reardon
Student Number: r0604915

Author: Danica Reardon
Student Number: r0604915

Python 3.5
"""

import os
from landmarks import load_landmarks, Landmarks
from tooth_model import Tooth

def main():
    directory = os.path.join(".", "_Data/Landmarks/")

    for num in range(1, 9):
        # 1.1 load landmarks
        landmarks = load_landmarks(directory, num, mirrored=False)
        tooth = Tooth(num)
        tooth.prepocess()


        # 1.2 process landmarks (Procrustes Analysis?)
        # mean_shape, aligned = procrustes(landmarks,incisor)
        #
        # # 1.3
        # PCA(mean_shape, aligned)

if __name__ == "__main__":
    main()
