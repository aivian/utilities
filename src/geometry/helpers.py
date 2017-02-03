""" A collection of geometrical helper things that don't fit in any other module
"""

import numpy as np

def unit_vector(index):
    """ Generate a unit vector

    Arguments:
        index: the index of the unit vector to generate

    Returns:
        n_hat: numpy (3,) array that is zeros except for a one in the position
            indicated by index
    """
    assert index < 3 and index > -1, "index must be between 0 and 2"
    n_hat = np.zeros((3,))
    n_hat[index] = 1.0
    return n_hat

def ring_index(this_array, inds):
    """ Index an array as a ring

    Takes a numpy array and indexes it as if it were a ring, for instance,
    given a 10 element array, ring_index(arr, 0) == ring_index(arr, 10)

    Arguments:
        this_array: numpy array
        inds: index or list of indices

    Returns:
        new_array: indexes out the desired items
    """

    assert isinstance(this_array, np.ndarray), "Array must be numpy array"
    if np.isscalar(this_array):
        return this_array
    if type(inds) is int:
        inds = [inds]
    inds = [i % len(this_array) for i in inds]
    return this_array[inds]

