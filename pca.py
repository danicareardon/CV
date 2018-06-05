import numpy as np
from procustes_analysis import procrustes
from landmarks import get_vectors, Landmarks
from matplotlib.mlab import PCA
from scipy import linalg as LA

def pca_code(landmarks):
    """ performing PCA to build an ASM

    Args:
        landmarks (list[Landmarks]) : all aligned landmarks for a tooth

    """
    # based off of Assignment 3
    # and https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-
    # vectors-from-sklearn-pca answer

    var_per = 0.98

    # covariance matrix
    X = get_vectors(landmarks)
    cov = np.cov(X, rowvar=0)

    # get the eigs
    evals, evecs = LA.eigh(cov)

    # sort by highest eigenvalue
    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:,idx]

    # choose number of eigenvectors for new dataset
    var = np.cumsum(evals)/np.sum(evals)
    i = np.argmax(var>=var_per)
    evecs = evecs[:,:i]

    reduced = np.dot(evecs.T,X.T).T
    return reduced, evals, evecs
