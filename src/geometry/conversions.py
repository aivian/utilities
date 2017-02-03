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
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3Stamped

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

def vector3_to_numpy(v3, ndmin=1):
    """ Convert a geometry_msgs/Vector3 to numpy array

    Arguments:
        v3: ROS geometry_msgs/Vector3 message

    Returns:
        np_vector: 3, numpy vector
    """
    #assert type(v3) is Vector3, "must input a Vector3!"
    assert hasattr(v3,'x') and hasattr(v3,'y') and hasattr(v3,'z'), "must input a Vector3-like structure!"
    return np.array([v3.x, v3.y, v3.z], ndmin=ndmin)

def numpy_to_vector3(np_vector):
    """ Convert a numpy array to a vector3

    Arguments:t.
        np_vector: 3, numpy array

    Returns:
        v3: Vector3 message
    """
    assert type(np_vector) is np.ndarray, "np_vector must be numpy array"
    #assert np_vector.shape == (3,), "must be 3, numpy array"
    if np_vector.shape == (2,):
        np_vector = np.append(np_vector, 0.0)
    v3 = Vector3()
    v3.x = np_vector[0]
    v3.y = np_vector[1]
    v3.z = np_vector[2]
    return v3

def numpy_to_vector3stamped(np_vector):
    """ Convert a numpy array to a vector3

    Arguments:t.
        np_vector: 3, numpy array

    Returns:
        v3: Vector3 message
    """
    assert type(np_vector) is np.ndarray, "np_vector must be numpy array"
    assert np_vector.shape == (3,), "must be 3, numpy array"
    v3 = Vector3Stamped()
    v3.x = np_vector[0]
    v3.y = np_vector[1]
    v3.z = np_vector[2]
    return v3

def point_to_numpy(v3, ndmin=1):
    """ Convert a geometry_msgs/Vector3 to numpy array

    Arguments:
        v3: ROS geometry_msgs/Vector3 message

    Returns:
        np_vector: 3, numpy vector
    """
    #assert type(v3) is Vector3, "must input a Vector3!"
    assert hasattr(v3,'x') and hasattr(v3,'y') and hasattr(v3,'z'), "must input a Vector3-like structure!"
    return np.array([v3.x, v3.y, v3.z], ndmin=ndmin)

def numpy_to_point(np_vector):
    """ Convert a numpy array to a vector3

    Arguments:
        np_vector: 3, numpy array

    Returns:
        v3: Vector3 message
    """
    assert type(np_vector) is np.ndarray, "np_vector must be numpy array"
    assert np_vector.shape == (3,), "must be 3, numpy array"
    v3 = Point()
    v3.x = np_vector[0]
    v3.y = np_vector[1]
    v3.z = np_vector[2]
    return v3

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
