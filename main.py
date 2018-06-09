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

    for num in range(1, 2):
        # 1.1 load landmarks
        landmarks = load_landmarks(directory, num, mirrored=False)
        tooth = Tooth(num)
        tooth.preprocess(landmarks)
        tooth.ASM(tooth.aligned)
        tooth.model_reconstruction(tooth.aligned)

if __name__ == "__main__":
    main()
