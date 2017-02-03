""" contains line related things
"""

import numpy as np
from math import sin, cos, sqrt, pi, fmod
from shapely.geometry import Point, LineString, LinearRing
import copy

def point_line_distance(pt, vertices, is_segment=True):
    """
    compute the minimum vector between a point and a line

    Arguments:
        pt: numpy 1x3 array of the point or a shapely point instance
        vertices: numpy nx3 array of points defining the end of the line segment
            or a shapely LineString or LinearRing instance
        is_segment: optional (defaults true), if false then the point are
            treated as defining an infinitely long line. False is only valid if
            line is a single segment.

    Return:
        r: vector from the point to the line or line segment
        x: numpy 2x3 array of the line segment
        i: the index of the beginning of the closest line segment
    """
    if type(pt) is Point:
        pt = np.array(pt.coords)

    if type(vertices) is LineString or type(vertices) is LinearRing:
        vertices = np.array(vertices.coords)

    if vertices.shape[0] > 2:
        assert is_segment, "only segment computation is valid for lines with\
            more than two points"
        # compute the point as the minimum of this function called recursively
        # over all line segments.
        min_mag_r = np.inf
        closest_ind = 0
        r = point_line_distance(pt, vertices[0:2], True)[0]
        for i in range(vertices.shape[0] - 1):
            test_r = point_line_distance(pt, vertices[i:i+2], True)[0]
            if min_mag_r > np.linalg.norm(test_r):
                min_mag_r = np.linalg.norm(test_r)
                r = copy.deepcopy(test_r)
                closest_ind = i
        return (r, vertices[closest_ind:closest_ind+2], closest_ind)

    # compute the distance to the infinite line
    line = vertices[-1] - vertices[0]
    line_hat = line/np.linalg.norm(line)
    r = (vertices[-1] - pt) - np.dot(vertices[-1] - pt, line_hat)*line_hat

    # return it if we aren't dealing with a segment
    if is_segment is False:
        return (r, vertices, 0)

    # check to see if both endpoints lie in the same direction (ie the point
    # does not fall between them...and the minimum distance is simply to one
    # endpoint, if so find the closer endpoint and return a vector to it
    r1 = vertices[-1] - pt
    r2 = vertices[0] - pt
    if np.array_equal(np.sign(r1 - r), np.sign(r2 - r)):
        if np.linalg.norm(r1) < np.linalg.norm(r2):
            return (r1, vertices, 0)
        else:
            return (r2, vertices, 0)
    return (r, vertices, 0)

def get_point_on_line(datum, vertices, distance, tol=1.0e-4):
    """ Find a point on a line a given distance from a datum.

    Given a line definition, a datum point and distance, find the point which
    lies on the line a given distance from the datum.

    Arguments:
        datum: numpy 3, array of the point or a shapely point instance
        vertices: numpy nx3 array of points defining the end of the line segment
            or a shapely LineString or LinearRing instance
        distance: floating point distance

    Returns:
        positive_point: the point on the line at the prescribed distance
            located in the positive line direction (defined as from the first
            to second vertex)
        negative_point: the point located along the negative line direction
    """

    if type(datum) is datum:
        datum = np.array(datum.coords)

    if type(vertices) is LineString or type(vertices) is LinearRing:
        vertices = np.array(vertices.coords)

    # compute some directions
    direction = vertices[1] - vertices[0]
    direction /= np.linalg.norm(direction)

    r = point_line_distance(datum, vertices, False)[0]

    line_distance = np.sqrt(-r.dot(r) + distance**2.0)

    positive_point = r + direction*line_distance
    negative_point = r - direction*line_distance

    return (positive_point, negative_point)

def line_intersections(Xa, ra, Xb ,rb):
    """ Compute the intersection point of two lines

    Arguments:
        Xa: a point on line a
        ra: direction along line a
        Xb: a point on line b
        rb: direction along line b

    Returns:
        ta: scale factor for ra (Xi = Xa + ta * ra)
        tb: scale factor for rb (Xi = Xb + tb * rb)
    """
    assert isinstance(Xa, np.ndarray), "Xa must be numpy array"
    assert isinstance(ra, np.ndarray), "ra must be numpy array"
    assert isinstance(Xb, np.ndarray), "Xb must be numpy array"
    assert isinstance(rb, np.ndarray), "rb must be numpy array"

    assert Xa.shape == (3,), "Xa must be (3,)"
    assert ra.shape == (3,), "ra must be (3,)"
    assert Xb.shape == (3,), "Xb must be (3,)"
    assert rb.shape == (3,), "rb must be (3,)"

    normal = np.cross(ra, rb)

    if np.linalg.norm(normal) < 1.0e-4:
        ta = np.inf
        tb = np.inf
        return (ta, tb)

    delta_X = Xb - Xa

    if np.linalg.norm(delta_X) < 1.0e-4:
        ta = 0.0
        tb = 0.0
        return (ta, tb)

    ta = np.cross(delta_X, rb)[2] / normal[2]
    tb = np.cross(delta_X, ra)[2] / normal[2]

    return (ta, tb)
