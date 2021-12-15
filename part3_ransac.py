import math

import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    num_samples = math.ceil(math.log(1 - prob_success, 1 - ind_prob_correct ** sample_size))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################    
    itermax = calculate_num_ransac_iterations(0.99, 9, 0.5)
    
    i = 0
    c = 0
    scope = 0.1
    n = len(matches_a)
    
    match1b = np.hstack([matches_b, np.ones([n, 1])])
    match1a = np.hstack([matches_a, np.ones([n, 1])])    
    while (i < min(200 * n, itermax)): 
        i += 1
        sample = np.random.choice(n, 8)
        pa = matches_a[sample]
        pb = matches_b[sample]
        tmpF = estimate_fundamental_matrix(pa, pb)     
        inlirs = []

        for j in range(len(matches_a)):
            t,q,w = match1a[j]
            r,s,o = match1b[j].dot(tmpF)            
            ou = (r * t + s * q + o) / math.sqrt(r**2 + s**2)

            if np.abs(ou) < scope: 
                inlirs.append(j)
        inlirs = np.array(inlirs)
        
        if len(inlirs) > c:
            c = len(inlirs)
            best_F, inliers_a, inliers_b = tmpF, matches_a[inlirs], matches_b[inlirs]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
