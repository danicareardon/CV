"""

 Class for exposing shape class variation using PCA

"""

import numpy as np
from procustes_analysis import procrustes

def PCA(mean_shape, aligned):
    #calculating the covariance for PCA
    aligned_shapes_vector = convert_landmarks_to_vectors(aligned_to_mean)
    covari = np.cov(aligned_shapes_vector, rowvar=0)

    #performing PCA
    eigen_values, eigen_vectors = np.linalg.eigh(covari)
    sortx = np.argsort(-eigen_values)
    eigen_values = eigen_values[sortx]
    eigen_vectors = eigen_vectors[:, sortx]

    vari = np.cumsum(eigen_values/np.sum(eigen_values))

    n_princ_comps, _ = extraction_of_components(vari > 0.99)
    n_princ_comps += 1

    P = []
    for i in range(0, n_princ_comps-1):
        P.append(np.sqrt(eigen_values[i]) * eigen_values[:, i])
    princ_modes = np.array(P).squeeze().T

    return princ_modes


def convert_landmarks_to_vectors(landmarks):
    """
    Method to convert a list of premade landmarks to vector format for PCA.
    """
    coord_array = []
    for mrks in landmarks:
        coord_array.append(mrks.get_vector())
    return np.array(coord_array)

#model building
def extraction_of_components(arr):
    for index, item in enumerate(arr):
        if item:
            return index,item
