import numpy as np
from procustes_analysis import procrustes
import landmarks as lms

def pca_code(landmarks):
    """ performing PCA to build an ASM

    Args:
        landmarks (list[Landmarks]) : all aligned landmarks for a tooth

    """
    # based off of Assignment 3
    # and https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-
    # vectors-from-sklearn-pca answer

    var_per = 0.98

    X = lms.get_vectors(landmarks)
    cov = np.cov(X, rowvar=0)

    evals, evecs = np.linalg.eigh(cov)

    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:,idx]

    var = np.cumsum(evals)/np.sum(evals)
    i = np.argmax(var>=var_per)
    evecs = evecs[:,:i+1]
    reduced = np.dot(evecs.T, X.T).T
    return reduced
