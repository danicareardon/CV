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
    X = get_vectors(landmarks)
    mu = X.mean(axis=0)
    X = X - mu

    var_per = 0.98

    # covariance matrix
    cov = np.cov(X, rowvar=0)

    # get the eigs
    evals, evecs = LA.eigh(cov)

    # sort by highest eigenvalue
    idx = np.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:,idx]

    # # choose number of eigenvectors for new dataset
    # var = np.cumsum(evals)/np.sum(evals)
    # i = np.argmax(var>=var_per)
    # evecs = evecs[:,:i]
    #
    # # reduced = np.dot(evecs.T,X.T).T
    return evals,evecs,mu

def num_eig(var_per,evals):
    """ chooses the number of eigenvalues
        Args:
            var_per (int) : 0 < var_per < 1.0
                the variance difference
            evals : (list) : the eigenvalues

        Returns:
            (int) : number of eigenvectors to choose
    """
    var = np.cumsum(evals)/np.sum(evals)
    return np.argmax(var>=var_per)
