""" A module for making iterables into a ring

This exists in geometry because the most immediate use for it I can think of
is to allow a sequence of waypoints to be indexed naturally without worrying
about whether I ran over the end of it (ie if I go one from the last I'd like
to run back to the first...). It might move later
"""

import numpy as np

def RingArray(object):
    """ A class to handle making an iterable into a ring
    """
    def __init__(self, X):
        """ Constructor

        Arguments:
            X: the values that will be iterable. Its type should be iterable

        Returns:
            object instance
        """
        super(RingArray, self).__init__()

        self._X = X
        self._i = 0

    def __getitem__(self, key):
        """ Get a value, wrap around the length...
        """
        i = mod(key, len(self._X))
        return self._X[i]

    def __setitem__(self, key, item):
        """ Set a value with a wrapped key
        """
        i = np.mod(key, len(self._X))
        self._X[i] = item

    def __iter__(self):
        """ Iterator
        """
        self._i = 0
        return self

    def next(self):
        """ do the iteration
        """
        if self._i > len(self._X):
            raise StopIteration
        else:
            self._i += 1
            return self[self._i]



