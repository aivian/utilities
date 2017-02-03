""" contains rotation related things
"""

import numpy as np
from math import sin, cos, sqrt, pi, fmod
#from shapely.geometry import Point, LineString, LinearRing

def xrot(angle):
    """
    generate a rotation matrix for an x rotation

    Arguments:
        angle: angle to rotate about the x axis in radians

    Returns:
        R: 3x3 numpy array defining the desired rotation matrix
    """
    R = np.zeros([3, 3])
    R[0, 0] = 1.0
    R[1, 1] = cos(angle)
    R[2, 2] = cos(angle)
    R[1, 2] = sin(angle)
    R[2, 1] = -sin(angle)
    return R

def yrot(angle):
    """
    generate a rotation matrix for an y rotation

    Arguments:
        angle: angle to rotate about the y axis in radians

    Returns:
        R: 3x3 numpy array defining the desired rotation matrix
    """
    R = np.zeros([3, 3])
    R[0, 0] = cos(angle)
    R[2, 2] = cos(angle)
    R[0, 2] = -sin(angle)
    R[2, 0] = sin(angle)
    R[1, 1] = 1.0
    return R

def zrot(angle):
    """
    generate a rotation matrix for an z rotation

    Arguments:
        angle: angle to rotate about the z axis in radians

    Returns:
        R: 3x3 numpy array defining the desired rotation matrix
    """
    R = np.zeros([3, 3])
    R[0, 0] = cos(angle)
    R[1, 1] = cos(angle)
    R[1, 0] = -sin(angle)
    R[0, 1] = sin(angle)
    R[2, 2] = 1.0
    return R

def rotate(axis, angle):
    """
    generates a rotation matrix for a specified angle/axis

    Can also be used to generate a generalized rotation matrix by passing a
    sequence of rotations in a list. The order of the list is interpreted such
    that element 0 defines the first rotation applied to a vector, 1 the second
    and so on...

    Arguments:
        axis: integer or string specifying axis, or a list of axes to rotate
            about. The axis/specification is detailed below:
            |____axis____|_valid symbols_|
            |      x     |   'x' or 1    |
            |      y     |   'y' or 2    |
            |      z     |   'z' or 3    |
        angle: angle or list of angles to rotate by, in radians

    Returns:
        R: 3x3 numpy array of the desired rotation
    """
    # for a sequence of rotations implement this function recursively
    if type(axis) is list:
        R = np.identity(3)
        for (ax, theta) in zip(axis, angle):
            R = np.dot(rotate(ax, theta), R)
        return R

    # define a dictionary of the functions and index to the desired one
    rot_dict = {1: xrot, 2: yrot, 3: zrot, 'x': xrot, 'y': yrot, 'z': zrot}
    return rot_dict[axis](angle)

def angle_difference(angle_1, angle_2):
    """
    compute the difference between two angles

    this takes care of the wraparound at -pi/pi in angles and delivers the
    minimum angular distance between two angles

    Arguments:
        angle_1: the first angle
        angle_2: the second angle which is subtracted from the first

    Returns:
        delta: the difference in two angles (angle_2 + delta = angle_1)
    """
    delta = angle_1 - angle_2
    delta = fmod((delta + pi), (2.0*pi)) - pi
    if delta < -pi:
        delta += 2.0*pi
    return delta
