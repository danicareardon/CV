"""Incisor Segmentation main function

Author: Danica Reardon
Student Number: r0604915

Author: Danica Reardon
Student Number: r0604915

Python 3.5
"""

import os
from landmarks import load_landmarks, Landmarks

def main():
    directory = os.path.join(".", "_Data/Landmarks/")

    for incisor in range(1,9):
        # 1.1 load landmarks
        landmarks = load_landmarks(directory, incisor, mirrored=True)


        # 1.2 process landmarks (Procrustes Analysis?)

        # 1.3






if __name__ == "__main__":
    main()
