"""
a collection of functions for computing geometric things, right now it contains

rotation matrices
closest point to a line segment or line
"""

import numpy as np
from math import sin
from math import cos
from math import sqrt
from math import pi

import warnings

def r2d(angle=1.0):
    """
    convert radian angles to degrees

    Arguments:
        angle: optional (defaults to 1 to return the conversion factor)

    Returns:
        angle_degrees
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' the numpy rad to degree function', DeprecationWarning)
    return np.rad2deg(angle)

def d2r(angle=1.0):
    """
    convert degree angles to radians

    Arguments:
        angle: optional (defaults to 1 to return the conversion factor)

    Returns:
        angle_radians
    """
    warnings.warn('This function has been deprecated, it is now just a call to'
        + ' the numpy degree to rad function', DeprecationWarning)
    return np.deg2rad(angle)

def to_unit_vector(this_vector):
    """ Convert a numpy vector to a unit vector

    Arguments:
        this_vector: a (3,) numpy array

    Returns:
        new_vector: a (3,) array with the same direction but unit length
    """
    norm = np.linalg.norm(this_vector)
    assert norm > 0.0, "vector norm must be greater than 0"
    if norm:
        return this_vector/np.linalg.norm(this_vector)
    else:
        return this_vector
