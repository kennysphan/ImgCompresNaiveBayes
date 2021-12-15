"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    x, y = np.mean(points, axis = 0)
    p1 = np.ones([len(points),3])
    p1[:,0:2] = points
    scaley = 1.0 / np.std((points - np.array([x, y]))[:,1])
    scalex = 1.0 / np.std((points - np.array([x, y]))[:,0])
    scalema = np.array([[scalex, 0 ,0], [0,scaley, 0], [0,0,1]])
    offsetmat = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])

    T = scalema.dot(offsetmat)
    pnormalize = p1.dot(T.T)
    points_normalized = pnormalize[:,0:2]
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T.dot(F_norm).dot(T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    anorm, t1 = normalize_points(points_a)
    bnorm, t2 = normalize_points(points_b)
    pA = np.zeros([len(points_a), 8]) 
    for i in range(anorm.shape[0]):
        ha, kx = anorm[i]
        hb, kt = bnorm[i]
        pA[i] = [ha * hb, kx * hb, hb, ha * kt, kx * kt, kt, ha, kx]

    tmp = -1 * np.ones(pA.shape[0])
    normF, re, ra, s = np.linalg.lstsq(pA, tmp, rcond = None)
    F = np.append(normF,1).reshape(3,3)
    ta, tus, h = np.linalg.svd(F)
    tus = np.diag(tus)
    tus[2,2] = 0
    normF = ta.dot(tus).dot(h)
    F = unnormalize_F(normF, t1, t2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
